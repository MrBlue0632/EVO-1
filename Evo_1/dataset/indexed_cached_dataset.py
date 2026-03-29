import io
import json
import logging
import mmap
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from .utils import normalize_minmax, pad_tensor

logger = logging.getLogger(__name__)
INDEXED_MANIFEST_SCHEMA_VERSION = 2


class _ShardReader:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def read(self, offset: int, size: int) -> bytes:
        end = offset + size
        if offset < 0 or size <= 0 or end > len(self.mm):
            raise ValueError(f"Invalid read range offset={offset} size={size} for shard={self.path}")
        return self.mm[offset:end]

    def close(self) -> None:
        self.mm.close()
        self.file.close()


class IndexedCachedLeRobotDataset(Dataset):
    def __init__(
        self,
        manifest_path: Union[str, Path],
        image_size: int = 448,
        use_augmentation: bool = False,
        augmentation_mode: str = "legacy_mix",
        augmentation_prob: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self._validate_manifest(manifest)

        self.manifest = manifest
        self.root = self.manifest_path.parent
        self.arm_to_embodiment_id = manifest["arm_to_embodiment_id"]
        self.arm2stats_dict = manifest["arm2stats_dict"]
        self.max_action_dim = int(manifest["max_action_dim"])
        self.max_state_dim = int(manifest["max_state_dim"])
        self.max_views = int(manifest["max_views"])
        self.image_size = int(image_size)
        self.use_augmentation = bool(use_augmentation)
        self.augmentation_mode = str(augmentation_mode).lower()
        self.augmentation_prob = min(max(float(augmentation_prob), 0.0), 1.0)
        if self.augmentation_mode not in {"legacy_mix", "always", "off"}:
            raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")

        index_path = self.root / manifest["index_file"]
        with open(index_path, "r", encoding="utf-8") as handle:
            index_obj = json.load(handle)
        self.records = index_obj["records"]
        if len(self.records) != int(manifest["num_samples"]):
            raise ValueError(
                f"Indexed cache mismatch: manifest num_samples={manifest['num_samples']} "
                f"but index has {len(self.records)} records"
            )

        self.shards = []
        for shard_name in manifest["shards"]:
            shard_path = self.root / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Indexed shard file not found: {shard_path}")
            self.shards.append(_ShardReader(shard_path))

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

    def _choose_image_transform(self):
        if (not self.use_augmentation) or self.augmentation_mode == "off":
            return self.basic_transform
        if self.augmentation_mode == "always":
            return self.aug_transform
        if torch.rand((), dtype=torch.float32).item() < self.augmentation_prob:
            return self.aug_transform
        return self.basic_transform

    def _validate_manifest(self, manifest: Dict[str, Any]) -> None:
        schema_version = int(manifest.get("schema_version", 0))
        if schema_version != INDEXED_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported indexed manifest schema_version={schema_version}, "
                f"expected {INDEXED_MANIFEST_SCHEMA_VERSION}"
            )
        required_fields = [
            "format",
            "export_key",
            "arm_to_embodiment_id",
            "arm2stats_dict",
            "max_action_dim",
            "max_state_dim",
            "max_views",
            "index_file",
            "shards",
            "num_samples",
        ]
        missing_fields = [field for field in required_fields if field not in manifest]
        if missing_fields:
            raise ValueError(f"Indexed manifest missing required fields: {missing_fields}")
        if manifest["format"] != "indexed_v2":
            raise ValueError(f"Unsupported indexed cache format: {manifest['format']}")

    def __len__(self) -> int:
        return len(self.records)

    def _decode_record(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        shard_idx = int(rec["shard"])
        offset = int(rec["offset"])
        size = int(rec["size"])
        if shard_idx < 0 or shard_idx >= len(self.shards):
            raise IndexError(f"Invalid shard index {shard_idx} for sample {idx}")
        raw = self.shards[shard_idx].read(offset, size)
        item = pickle.loads(raw)
        if not isinstance(item, dict):
            raise TypeError(f"Invalid indexed sample type for {idx}: {type(item)!r}")
        return item

    def _load_images(self, encoded_images: List[bytes]) -> Tuple[torch.Tensor, torch.Tensor]:
        transform = self._choose_image_transform()
        images = []
        for image_bytes in encoded_images:
            with Image.open(io.BytesIO(image_bytes)) as image:
                images.append(transform(image.convert("RGB")))
        if len(images) > self.max_views:
            raise ValueError(f"Cached sample has {len(images)} views, exceeds max_views {self.max_views}")

        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        while len(images) < self.max_views:
            images.append(torch.zeros(3, self.image_size, self.image_size))
        return torch.stack(images), image_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._decode_record(idx)
        arm_key = sample["arm_key"]
        dataset_key = sample["dataset_key"]
        norm_stats = self.arm2stats_dict[arm_key]

        encoded_images = sample.get("images_encoded")
        if encoded_images is None:
            encoded_images = sample.get("images_jpeg")
        if encoded_images is None:
            raise KeyError("Indexed sample is missing both images_encoded and images_jpeg fields.")

        images, image_mask = self._load_images(encoded_images)
        state = torch.tensor(sample["state"], dtype=torch.float32)
        state_min = torch.tensor(norm_stats["observation.state"]["min"], dtype=torch.float32)
        state_max = torch.tensor(norm_stats["observation.state"]["max"], dtype=torch.float32)
        state = normalize_minmax(state, state_min, state_max)
        state_padded, state_mask = pad_tensor(state, self.max_state_dim)

        action = torch.tensor(np.asarray(sample["action"]), dtype=torch.float32)
        action_min = torch.tensor(norm_stats["action"]["min"], dtype=torch.float32)
        action_max = torch.tensor(norm_stats["action"]["max"], dtype=torch.float32)
        action = normalize_minmax(action, action_min.unsqueeze(0), action_max.unsqueeze(0))
        action_padded, action_mask = pad_tensor(action, self.max_action_dim)

        return {
            "images": images,
            "image_mask": image_mask,
            "prompt": sample.get("prompt") or "",
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(self.arm_to_embodiment_id[arm_key], dtype=torch.long),
            "dataset_key": dataset_key,
        }

    def __del__(self):
        for shard in getattr(self, "shards", []):
            try:
                shard.close()
            except Exception:
                pass
