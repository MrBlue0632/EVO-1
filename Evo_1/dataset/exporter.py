import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from tqdm.auto import tqdm

from .lerobot_dataset_pretrain_mp import LeRobotDataset
from .utils import atomic_write_json, dataset_cache_root, fingerprint_payload

logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1


def build_export_key(config: Dict[str, Any], dataset_config: Dict[str, Any]) -> str:
    return fingerprint_payload({
        'dataset_config': dataset_config,
        'image_size': config.get('image_size', 448),
        'horizon': config.get('horizon', 50),
        'max_samples_per_file': config.get('max_samples_per_file'),
        'cache_schema': 3,
    })


def _save_images(image_dir: Path, sample_idx: int, images: List[Image.Image]) -> List[str]:
    relative_paths = []
    sample_dir = image_dir / f'{sample_idx:08d}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    for view_idx, image in enumerate(images):
        path = sample_dir / f'view_{view_idx}.jpg'
        image.save(path, format='JPEG', quality=95)
        relative_paths.append(str(path.relative_to(image_dir.parent)))
    return relative_paths


def export_cached_dataset(config: Dict[str, Any], dataset_config: Dict[str, Any]) -> Path:
    cache_root = dataset_cache_root(config.get('cache_dir'))
    export_key = build_export_key(config, dataset_config)
    export_root = cache_root / 'exports' / export_key
    manifest_path = export_root / 'manifest.json'
    if manifest_path.exists():
        logger.info('Reusing cached dataset manifest at %s', manifest_path)
        return manifest_path

    export_root.mkdir(parents=True, exist_ok=True)
    image_dir = export_root / 'media'
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        config=dataset_config,
        image_size=config.get('image_size', 448),
        max_samples_per_file=config.get('max_samples_per_file'),
        action_horizon=config.get('horizon', 50),
        binarize_gripper=config.get('binarize_gripper', False),
        cache_dir=config.get('cache_dir'),
        use_augmentation=False,
        num_cache_workers=config.get('dataset_num_workers'),
    )

    samples = []
    for sample_idx in tqdm(range(len(dataset)), desc='Export cached dataset'):
        raw_item = dataset._load_raw_item(sample_idx)
        image_paths = _save_images(image_dir, sample_idx, raw_item['frames'])
        samples.append({
            'arm_key': raw_item['arm_key'],
            'dataset_key': raw_item['dataset_key'],
            'prompt': raw_item['prompt'],
            'state': raw_item['state'],
            'action': raw_item['action'],
            'image_paths': image_paths,
        })

    manifest = {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'export_key': export_key,
        'arm_to_embodiment_id': dataset.arm_to_embodiment_id,
        'arm2stats_dict': dataset.arm2stats_dict,
        'max_action_dim': dataset.max_action_dim,
        'max_state_dim': dataset.max_state_dim,
        'max_views': dataset.max_views,
        'samples': samples,
    }
    atomic_write_json(manifest_path, manifest)
    logger.info('Exported cached dataset manifest to %s with %s samples', manifest_path, len(samples))
    return manifest_path
