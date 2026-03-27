import io
import json
import logging
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

try:
    from .utils import atomic_write_bytes, atomic_write_json, dataset_cache_root, fingerprint_payload, normalize_minmax, pad_tensor
except ImportError:
    from dataset.utils import atomic_write_bytes, atomic_write_json, dataset_cache_root, fingerprint_payload, normalize_minmax, pad_tensor

logger = logging.getLogger(__name__)


def compute_lerobot_normalization_stats_from_minmax(jsonl_path: Path) -> Dict[str, Dict[str, List[float]]]:
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    with open(jsonl_path, 'r', encoding='utf-8') as handle:
        for line in tqdm(handle, desc=f'Extracting min/max from {jsonl_path.name}'):
            obj = json.loads(line)
            stats = obj.get('stats', {})
            try:
                state_mins.append(stats['observation.state']['min'])
                state_maxs.append(stats['observation.state']['max'])
                action_mins.append(stats['action']['min'])
                action_maxs.append(stats['action']['max'])
            except KeyError as exc:
                logger.warning('Skipping abnormal stats line in %s: %s', jsonl_path, exc)

    if not state_mins or not action_mins:
        raise ValueError(f'No valid normalization stats found in {jsonl_path}')

    return {
        'observation.state': {
            'min': np.min(np.asarray(state_mins), axis=0).tolist(),
            'max': np.max(np.asarray(state_maxs), axis=0).tolist(),
        },
        'action': {
            'min': np.min(np.asarray(action_mins), axis=0).tolist(),
            'max': np.max(np.asarray(action_maxs), axis=0).tolist(),
        },
    }


def merge_lerobot_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, List[float]]]:
    if not stats_list:
        raise ValueError('stats_list must not be empty')
    state_mins = [np.array(d['observation.state']['min']) for d in stats_list]
    state_maxs = [np.array(d['observation.state']['max']) for d in stats_list]
    action_mins = [np.array(d['action']['min']) for d in stats_list]
    action_maxs = [np.array(d['action']['max']) for d in stats_list]
    return {
        'observation.state': {
            'min': np.min(np.stack(state_mins), axis=0).tolist(),
            'max': np.max(np.stack(state_maxs), axis=0).tolist(),
        },
        'action': {
            'min': np.min(np.stack(action_mins), axis=0).tolist(),
            'max': np.max(np.stack(action_maxs), axis=0).tolist(),
        },
    }


def _dump_pickle_bytes(payload: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(payload, buffer)
    return buffer.getvalue()


def _default_view_map() -> Dict[str, str]:
    return {
        'image_1': 'observation.images.image_1',
        'image_2': 'observation.images.image_2',
        'image_3': 'observation.images.image_3',
    }


def _build_cache_key(dataset_path: Path, action_horizon: int, max_samples_per_file: Optional[int]) -> str:
    return fingerprint_payload({
        'dataset_path': str(dataset_path.resolve()),
        'action_horizon': action_horizon,
        'max_samples_per_file': max_samples_per_file,
        'cache_schema': 2,
    })


def _process_parquet_file_worker(args):
    (
        parquet_path,
        arm_name,
        dataset_name,
        dataset_config,
        dataset_path,
        task_mapping,
        action_horizon,
        max_samples_per_file,
        cache_dir,
    ) = args

    try:
        view_map = dataset_config.get('view_map') or _default_view_map()
        df = pd.read_parquet(parquet_path)

        last_row = df.iloc[-1:]
        padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)
        if max_samples_per_file is not None:
            df = df.head(max_samples_per_file)

        episode_files = []
        for index in range(len(df) - action_horizon + 1):
            start_idx = index
            end_idx = index + action_horizon
            cache_subdir = cache_dir / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem
            cache_filepath = cache_subdir / f'{start_idx}_{end_idx}.pkl'
            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                continue

            sub_df = df.iloc[index:index + action_horizon]
            base_video_path = dataset_path / 'videos' / parquet_path.parent.name
            video_paths = {}
            for view_key, view_folder in view_map.items():
                full_path = base_video_path / view_folder / f'{parquet_path.stem}.mp4'
                if full_path.exists():
                    video_paths[view_key] = str(full_path)

            task_index = sub_df.iloc[0].get('task_index')
            prompt = task_mapping.get(task_index, '') if task_index is not None else ''
            episode = {
                'arm_key': arm_name,
                'dataset_key': dataset_name,
                'prompt': prompt,
                'state': sub_df.iloc[0].get('observation.state'),
                'action': [row['action'] for _, row in sub_df.iterrows()],
                'video_paths': video_paths,
                'timestamp': sub_df.iloc[0].get('timestamp'),
            }
            atomic_write_bytes(cache_filepath, _dump_pickle_bytes(episode))
            episode_files.append(str(cache_filepath))
        return episode_files, None
    except Exception as exc:
        return [], f'Error processing file {parquet_path}: {exc}'


class LeRobotDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Optional[int] = None,
        video_backend: str = 'av',
        action_horizon: int = 50,
        video_backend_kwargs: Optional[Dict[str, Any]] = None,
        binarize_gripper: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        use_augmentation: bool = False,
        num_cache_workers: Optional[int] = None,
    ):
        self.config = config
        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: index for index, key in enumerate(sorted_datasets)}
        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']
        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation
        self.cache_root = dataset_cache_root(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.data: List[str] = []
        self.arm2stats_dict: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}
        self.num_cache_workers = num_cache_workers

        self._load_metadata()
        self._load_trajectories()

        self.basic_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor(),
        ])

    def _stats_cache_path(self, dataset_path: Path) -> Path:
        cache_key = _build_cache_key(dataset_path, self.action_horizon, self.max_samples_per_file)
        return self.cache_root / 'stats' / f'{dataset_path.name}_{cache_key}.json'

    def _load_or_compute_stats(self, dataset_path: Path) -> Dict[str, Dict[str, List[float]]]:
        source_stats_path = dataset_path / 'meta' / 'stats.json'
        if source_stats_path.exists():
            with open(source_stats_path, 'r', encoding='utf-8') as handle:
                return json.load(handle)

        stats_path = dataset_path / 'meta' / 'episodes_stats.jsonl'
        if not stats_path.exists():
            raise FileNotFoundError(f'normalization stats file not found: {stats_path}')

        cache_path = self._stats_cache_path(dataset_path)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as handle:
                return json.load(handle)

        stats = compute_lerobot_normalization_stats_from_minmax(stats_path)
        atomic_write_json(cache_path, stats)
        return stats

    def _load_metadata(self):
        self.tasks: Dict[str, Dict[str, Dict[int, str]]] = {}
        for arm_name, arm_config in self.config['data_groups'].items():
            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = Path(dataset_config['path']).expanduser().resolve()
                tasks_path = dataset_path / 'meta' / 'tasks.jsonl'
                if not tasks_path.exists():
                    raise FileNotFoundError(f'tasks file not found: {tasks_path}')
                dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict('records')
                task_index_to_task = {
                    task_obj['task_index']: task_obj['task']
                    for task_obj in dataset_tasks
                    if 'task_index' in task_obj and 'task' in task_obj
                }
                self.tasks[arm_name][dataset_name] = task_index_to_task
                norm_arm_list.append(self._load_or_compute_stats(dataset_path))
            self.arm2stats_dict[arm_name] = merge_lerobot_stats(norm_arm_list)

    def _load_trajectories(self):
        parquet_process_units = []
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = Path(dataset_config.get('path')).expanduser().resolve()
                if not dataset_path.exists():
                    raise FileNotFoundError(f'Dataset path not found: {dataset_path}')
                parquet_files = sorted(dataset_path.glob('data/*/*.parquet'))
                cache_key = _build_cache_key(dataset_path, self.action_horizon, self.max_samples_per_file)
                dataset_cache_dir = self.cache_root / 'episodes' / cache_key
                task_mapping = self.tasks[arm_name][dataset_name]
                for parquet_path in parquet_files:
                    parquet_process_units.append((
                        parquet_path,
                        arm_name,
                        dataset_name,
                        dataset_config,
                        dataset_path,
                        task_mapping,
                        self.action_horizon,
                        self.max_samples_per_file,
                        dataset_cache_dir,
                    ))

        if not parquet_process_units:
            raise ValueError('No parquet files found for configured datasets')

        num_processes = self.num_cache_workers or min(max(mp.cpu_count() // 2, 1), len(parquet_process_units), 8)
        total_episodes = 0
        with mp.Pool(processes=num_processes) as pool:
            with tqdm(total=len(parquet_process_units), desc='Processing parquet files to cache') as progress:
                for episode_files, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logger.error(error)
                    else:
                        self.data.extend(episode_files)
                        total_episodes += len(episode_files)
                    progress.set_postfix({'episodes': total_episodes})
                    progress.update(1)

        if not self.data:
            raise ValueError('Dataset cache build produced zero episodes')
        logger.info('Prepared %s cached episodes under %s', len(self.data), self.cache_root)

    def _load_video_frame(self, video_paths: Dict[str, str], timestamp: float) -> List[Image.Image]:
        if not video_paths:
            raise ValueError('video_paths is empty for sample')

        frames = []
        for _, path in sorted(video_paths.items()):
            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f'video file not found: {path_obj}')

            if self.video_backend == 'decord':
                import decord

                ctx_name = self.video_backend_kwargs.get('ctx', 'cpu')
                ctx = decord.cpu(0) if ctx_name == 'cpu' else decord.gpu(0)
                reader = decord.VideoReader(str(path_obj), ctx=ctx)
                fps = reader.get_avg_fps()
                if fps is None or np.isnan(fps):
                    raise ValueError(f'Unable to read FPS from {path_obj}')
                frame_idx = min(int(timestamp * fps), len(reader) - 1)
                frame = reader[frame_idx].asnumpy()
                frames.append(Image.fromarray(frame))
                continue

            if self.video_backend == 'av':
                import av

                with av.open(str(path_obj)) as container:
                    selected_frame = None
                    for frame in container.decode(video=0):
                        if frame.time is not None and frame.time >= timestamp:
                            selected_frame = frame
                            break
                    if selected_frame is None:
                        raise ValueError(f'No frame found at timestamp {timestamp} for {path_obj}')
                    frames.append(Image.fromarray(selected_frame.to_ndarray(format='rgb24')))
                continue

            raise NotImplementedError(f'Video backend {self.video_backend} not implemented')
        return frames

    def __len__(self):
        return len(self.data)

    def _load_cached_episode(self, idx: int) -> Dict[str, Any]:
        cache_filepath = Path(self.data[idx])
        try:
            with open(cache_filepath, 'rb') as handle:
                item = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(f'Failed to load cache file {cache_filepath}: {exc}') from exc
        if not isinstance(item, dict):
            raise TypeError(f'Invalid episode cache type at {cache_filepath}: expected dict, got {type(item)!r}')
        item['_cache_filepath'] = str(cache_filepath)
        return item

    def _load_raw_item(self, idx: int) -> Dict[str, Any]:
        item = self._load_cached_episode(idx)
        cache_filepath = Path(item['_cache_filepath'])

        arm_key = item.get('arm_key')
        dataset_key = item.get('dataset_key')
        if arm_key is None or dataset_key is None:
            raise ValueError(f'missing arm_key/dataset_key in {cache_filepath}')

        frames = self._load_video_frame(item.get('video_paths', {}), item.get('timestamp'))
        state = item.get('state')
        action = item.get('action')
        if state is None:
            raise ValueError(f'missing observation.state in {cache_filepath}')
        if action is None:
            raise ValueError(f'missing action in {cache_filepath}')
        if not isinstance(action, list) or len(action) == 0:
            raise ValueError(f'invalid action sequence in {cache_filepath}: expected non-empty list')

        return {
            'arm_key': arm_key,
            'dataset_key': dataset_key,
            'prompt': item.get('prompt') or '',
            'state': state,
            'action': action,
            'frames': frames,
            'cache_filepath': str(cache_filepath),
        }

    def _build_training_item(self, raw_item: Dict[str, Any]) -> Dict[str, Any]:
        arm_key = raw_item['arm_key']
        dataset_key = raw_item['dataset_key']
        cache_filepath = raw_item.get('cache_filepath', '<unknown>')
        embodiment_id = self.arm_to_embodiment_id[arm_key]
        frames = raw_item['frames']
        images = [self.aug_transform(img) if self.use_augmentation else self.basic_transform(img) for img in frames]

        if len(images) > self.max_views:
            raise ValueError(f'Number of views {len(images)} exceeds max_views {self.max_views}')

        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        while len(images) < self.max_views:
            dummy_image = torch.zeros(3, self.image_size, self.image_size) if not images else torch.zeros_like(images[0])
            images.append(dummy_image)
        images = torch.stack(images)

        try:
            norm_stats = self.arm2stats_dict[arm_key]
        except KeyError as exc:
            raise KeyError(f'Normalization stats not found for arm_key={arm_key} dataset_key={dataset_key}') from exc

        state = torch.tensor(raw_item['state'], dtype=torch.float32)
        state_min = torch.tensor(norm_stats['observation.state']['min'], dtype=torch.float32)
        state_max = torch.tensor(norm_stats['observation.state']['max'], dtype=torch.float32)
        state = normalize_minmax(state, state_min, state_max)
        state_padded, state_mask = pad_tensor(state, self.max_state_dim)

        action = torch.from_numpy(np.stack(raw_item['action'])).float()
        action_min = torch.tensor(norm_stats['action']['min'], dtype=torch.float32)
        action_max = torch.tensor(norm_stats['action']['max'], dtype=torch.float32)
        action = normalize_minmax(action, action_min.unsqueeze(0), action_max.unsqueeze(0))
        action_padded, action_mask = pad_tensor(action, self.max_action_dim)

        prompt = raw_item['prompt'] or ''
        return {
            'images': images,
            'image_mask': image_mask,
            'prompt': prompt,
            'state': state_padded.to(dtype=torch.bfloat16),
            'state_mask': state_mask,
            'action': action_padded.to(dtype=torch.bfloat16),
            'action_mask': action_mask,
            'embodiment_id': torch.tensor(embodiment_id, dtype=torch.long),
            'dataset_key': dataset_key,
            'cache_filepath': cache_filepath,
        }

    def __getitem__(self, idx):
        raw_item = self._load_raw_item(idx)
        return self._build_training_item(raw_item)
