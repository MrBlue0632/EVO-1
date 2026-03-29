from pathlib import Path

import logging
import yaml
from torch.utils.data import DataLoader

try:
    from .cached_dataset import CachedLeRobotDataset
    from .exporter import build_export_key, export_cached_dataset
    from .lerobot_dataset_pretrain_mp import LeRobotDataset
    from .utils import collate_batch
except ImportError:
    from dataset.cached_dataset import CachedLeRobotDataset
    from dataset.exporter import build_export_key, export_cached_dataset
    from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset
    from dataset.utils import collate_batch

logger = logging.getLogger(__name__)


def load_dataset_config(config: dict) -> dict:
    dataset_config_path = config.get('dataset_config_path')
    if not dataset_config_path:
        raise ValueError('dataset_config_path must be provided')
    with open(Path(dataset_config_path).expanduser(), 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _resolve_manifest_path(config: dict, dataset_config: dict):
    cache_dir = config.get('cache_dir')
    from .utils import dataset_cache_root
    cache_root = dataset_cache_root(cache_dir)
    export_key = build_export_key(config, dataset_config)
    return cache_root / 'exports' / export_key / 'manifest.json'


def build_cached_dataset(config: dict):
    dataset_config = load_dataset_config(config)
    manifest_path = _resolve_manifest_path(config, dataset_config)
    logger.info('Dataset mode=cached export_key=%s', manifest_path.parent.name)
    if not manifest_path.exists():
        if config.get('auto_export_cached_dataset', True):
            manifest_path = export_cached_dataset(config, dataset_config)
        else:
            raise FileNotFoundError(f'Cached dataset manifest not found: {manifest_path}')
    return CachedLeRobotDataset(
        manifest_path,
        image_size=config.get('image_size', 448),
        use_augmentation=config.get('use_augmentation', False),
    )


def build_dataset(config: dict):
    dataset_type = config.get('dataset_type', 'lerobot')
    if dataset_type != 'lerobot':
        raise ValueError(f'Unknown dataset_type: {dataset_type}')

    if config.get('use_cached_dataset', True):
        return build_cached_dataset(config)

    logger.info('Dataset mode=legacy')
    dataset_config = load_dataset_config(config)
    return LeRobotDataset(
        config=dataset_config,
        image_size=config.get('image_size', 448),
        max_samples_per_file=config.get('max_samples_per_file'),
        action_horizon=config.get('horizon', 50),
        binarize_gripper=config.get('binarize_gripper', False),
        cache_dir=config.get('cache_dir'),
        use_augmentation=config.get('use_augmentation', False),
        num_cache_workers=config.get('dataset_num_workers'),
    )


def build_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = int(config.get('batch_size', 8))
    num_workers = int(config.get('num_workers', 8))
    pin_memory = bool(config.get('pin_memory', True))
    persistent_workers = bool(config.get('persistent_workers', num_workers > 0))
    prefetch_factor = config.get('prefetch_factor', 4)
    loader_kwargs = {}
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = int(prefetch_factor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
        collate_fn=collate_batch,
        **loader_kwargs,
    )
