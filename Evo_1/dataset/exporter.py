import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .lerobot_dataset_pretrain_mp import LeRobotDataset
from .utils import atomic_write_json, dataset_cache_root, fingerprint_payload

logger = logging.getLogger(__name__)
LEGACY_MANIFEST_SCHEMA_VERSION = 1
INDEXED_MANIFEST_SCHEMA_VERSION = 2


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_export_key(config: Dict[str, Any], dataset_config: Dict[str, Any]) -> str:
    cache_format = str(config.get("cache_format", "indexed_v2")).lower()
    cache_image_codec = str(config.get("cache_image_codec", "png")).lower()
    return fingerprint_payload({
        'dataset_config': dataset_config,
        'image_size': config.get('image_size', 448),
        'horizon': config.get('horizon', 50),
        'max_samples_per_file': config.get('max_samples_per_file'),
        'cache_format': cache_format,
        'cache_image_codec': cache_image_codec,
        'cache_shard_size_mb': int(config.get('cache_shard_size_mb', 512)),
        'cache_schema': 5,
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


def _encode_image(image: Image.Image, codec: str = "png", quality: int = 95) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    rgb_image = image.convert("RGB")
    if codec == "png":
        rgb_image.save(buffer, format="PNG")
    elif codec == "jpeg":
        rgb_image.save(buffer, format="JPEG", quality=quality)
    else:
        raise ValueError(f"Unsupported cache_image_codec: {codec}")
    return buffer.getvalue()


class _IndexedShardWriter:
    def __init__(self, export_root: Path, shard_size_mb: int):
        self.export_root = export_root
        self.shard_size_bytes = max(16 * 1024 * 1024, int(shard_size_mb) * 1024 * 1024)
        self.records: List[Dict[str, int]] = []
        self.shard_files: List[str] = []
        self.shard_idx = 0
        self.current_size = 0
        self.file = None
        self._open_shard()

    def _open_shard(self) -> None:
        shard_name = f"data-{self.shard_idx:05d}.bin"
        shard_path = self.export_root / shard_name
        self.file = open(shard_path, "wb")
        self.shard_files.append(shard_name)
        self.current_size = 0

    def _rotate_if_needed(self, payload_size: int) -> None:
        if self.current_size == 0:
            return
        if self.current_size + payload_size <= self.shard_size_bytes:
            return
        self.file.close()
        self.shard_idx += 1
        self._open_shard()

    def add_record(self, record: Dict[str, Any]) -> None:
        payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        self._rotate_if_needed(len(payload))
        offset = self.current_size
        self.file.write(payload)
        self.records.append({
            "shard": self.shard_idx,
            "offset": offset,
            "size": len(payload),
        })
        self.current_size += len(payload)

    def finalize(self) -> Dict[str, Any]:
        self.file.close()
        return {
            "records": self.records,
            "shards": self.shard_files,
        }


def _build_dataset_for_export(config: Dict[str, Any], dataset_config: Dict[str, Any]) -> LeRobotDataset:
    return LeRobotDataset(
        config=dataset_config,
        image_size=config.get("image_size", 448),
        max_samples_per_file=config.get("max_samples_per_file"),
        action_horizon=config.get("horizon", 50),
        binarize_gripper=config.get("binarize_gripper", False),
        cache_dir=config.get("cache_dir"),
        use_augmentation=False,
        num_cache_workers=config.get("dataset_num_workers"),
        episode_cache_mode="index",
    )


def _export_cached_dataset_legacy(config: Dict[str, Any], dataset_config: Dict[str, Any], export_root: Path, export_key: str) -> Path:
    manifest_path = export_root / "manifest.json"
    image_dir = export_root / "media"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset_for_export(config, dataset_config)

    samples = []
    for sample_idx in tqdm(range(len(dataset)), desc="Export cached dataset (legacy)"):
        raw_item = dataset._load_raw_item(sample_idx)
        image_paths = _save_images(image_dir, sample_idx, raw_item["frames"])
        samples.append({
            "arm_key": raw_item["arm_key"],
            "dataset_key": raw_item["dataset_key"],
            "prompt": raw_item["prompt"],
            "state": _to_jsonable(raw_item["state"]),
            "action": _to_jsonable(raw_item["action"]),
            "image_paths": image_paths,
        })

    manifest = {
        "schema_version": LEGACY_MANIFEST_SCHEMA_VERSION,
        "export_key": export_key,
        "arm_to_embodiment_id": dataset.arm_to_embodiment_id,
        "arm2stats_dict": dataset.arm2stats_dict,
        "max_action_dim": dataset.max_action_dim,
        "max_state_dim": dataset.max_state_dim,
        "max_views": dataset.max_views,
        "samples": samples,
    }
    atomic_write_json(manifest_path, _to_jsonable(manifest))
    return manifest_path


def _export_cached_dataset_indexed(config: Dict[str, Any], dataset_config: Dict[str, Any], export_root: Path, export_key: str) -> Path:
    manifest_path = export_root / "manifest.json"
    index_path = export_root / "index.json"
    dataset = _build_dataset_for_export(config, dataset_config)
    writer = _IndexedShardWriter(export_root, shard_size_mb=int(config.get("cache_shard_size_mb", 512)))
    image_codec = str(config.get("cache_image_codec", "png")).lower()
    if image_codec not in {"png", "jpeg"}:
        raise ValueError(f"Unsupported cache_image_codec: {image_codec}")

    for sample_idx in tqdm(range(len(dataset)), desc="Export cached dataset (indexed_v2)"):
        raw_item = dataset._load_raw_item(sample_idx)
        encoded_images = [_encode_image(image, codec=image_codec) for image in raw_item["frames"]]
        writer.add_record({
            "arm_key": raw_item["arm_key"],
            "dataset_key": raw_item["dataset_key"],
            "prompt": raw_item["prompt"],
            "state": _to_jsonable(raw_item["state"]),
            "action": _to_jsonable(raw_item["action"]),
            "image_codec": image_codec,
            "images_encoded": encoded_images,
        })

    index_obj = writer.finalize()
    atomic_write_json(index_path, _to_jsonable({"records": index_obj["records"]}))
    manifest = {
        "schema_version": INDEXED_MANIFEST_SCHEMA_VERSION,
        "format": "indexed_v2",
        "export_key": export_key,
        "arm_to_embodiment_id": dataset.arm_to_embodiment_id,
        "arm2stats_dict": dataset.arm2stats_dict,
        "max_action_dim": dataset.max_action_dim,
        "max_state_dim": dataset.max_state_dim,
        "max_views": dataset.max_views,
        "num_samples": len(index_obj["records"]),
        "image_codec": image_codec,
        "index_file": "index.json",
        "shards": index_obj["shards"],
    }
    atomic_write_json(manifest_path, _to_jsonable(manifest))
    return manifest_path


def export_cached_dataset(config: Dict[str, Any], dataset_config: Dict[str, Any]) -> Path:
    cache_root = dataset_cache_root(config.get('cache_dir'))
    export_key = build_export_key(config, dataset_config)
    export_root = cache_root / 'exports' / export_key
    manifest_path = export_root / 'manifest.json'
    if manifest_path.exists():
        logger.info('Reusing cached dataset manifest at %s', manifest_path)
        return manifest_path

    export_root.mkdir(parents=True, exist_ok=True)
    cache_format = str(config.get("cache_format", "indexed_v2")).lower()
    if cache_format == "legacy":
        manifest_path = _export_cached_dataset_legacy(config, dataset_config, export_root, export_key)
    elif cache_format == "indexed_v2":
        manifest_path = _export_cached_dataset_indexed(config, dataset_config, export_root, export_key)
    else:
        raise ValueError(f"Unknown cache_format: {cache_format}")
    logger.info("Exported cached dataset manifest to %s", manifest_path)
    return manifest_path
