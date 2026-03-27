import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from .utils import normalize_minmax, pad_tensor

logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1


class CachedLeRobotDataset(Dataset):

    def __init__(
        self,
        manifest_path: Union[str, Path],
        image_size: int = 448,
        use_augmentation: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with open(self.manifest_path, 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        self._validate_manifest(manifest)

        self.manifest = manifest
        self.root = self.manifest_path.parent
        self.samples = manifest['samples']
        self.arm_to_embodiment_id = manifest['arm_to_embodiment_id']
        self.arm2stats_dict = manifest['arm2stats_dict']
        self.max_action_dim = manifest['max_action_dim']
        self.max_state_dim = manifest['max_state_dim']
        self.max_views = manifest['max_views']
        self.image_size = image_size
        self.use_augmentation = use_augmentation

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

    def _validate_manifest(self, manifest: Dict[str, Any]) -> None:
        schema_version = int(manifest.get('schema_version', 0))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f'Unsupported manifest schema_version={schema_version}, expected {MANIFEST_SCHEMA_VERSION}'
            )

        required_fields = [
            'export_key',
            'samples',
            'arm_to_embodiment_id',
            'arm2stats_dict',
            'max_action_dim',
            'max_state_dim',
            'max_views',
        ]
        missing_fields = [field for field in required_fields if field not in manifest]
        if missing_fields:
            raise ValueError(f'Manifest missing required fields: {missing_fields}')

        if not isinstance(manifest['samples'], list):
            raise TypeError('Manifest field "samples" must be a list')
        if len(manifest['samples']) == 0:
            raise ValueError('Manifest contains zero samples')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_images(self, image_paths: List[str]) -> torch.Tensor:
        transform = self.aug_transform if self.use_augmentation else self.basic_transform
        images = []
        for image_path in image_paths:
            path = self.root / image_path
            if not path.exists():
                raise FileNotFoundError(f'Cached image not found: {path}')
            with Image.open(path) as image:
                images.append(transform(image.convert('RGB')))

        if len(images) > self.max_views:
            raise ValueError(f'Cached sample has {len(images)} views, exceeds max_views {self.max_views}')

        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        while len(images) < self.max_views:
            images.append(torch.zeros(3, self.image_size, self.image_size))
        return torch.stack(images), image_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        arm_key = sample['arm_key']
        dataset_key = sample['dataset_key']
        norm_stats = self.arm2stats_dict[arm_key]

        images, image_mask = self._load_images(sample['image_paths'])
        state = torch.tensor(sample['state'], dtype=torch.float32)
        state_min = torch.tensor(norm_stats['observation.state']['min'], dtype=torch.float32)
        state_max = torch.tensor(norm_stats['observation.state']['max'], dtype=torch.float32)
        state = normalize_minmax(state, state_min, state_max)
        state_padded, state_mask = pad_tensor(state, self.max_state_dim)

        action = torch.tensor(np.asarray(sample['action']), dtype=torch.float32)
        action_min = torch.tensor(norm_stats['action']['min'], dtype=torch.float32)
        action_max = torch.tensor(norm_stats['action']['max'], dtype=torch.float32)
        action = normalize_minmax(action, action_min.unsqueeze(0), action_max.unsqueeze(0))
        action_padded, action_mask = pad_tensor(action, self.max_action_dim)

        return {
            'images': images,
            'image_mask': image_mask,
            'prompt': sample.get('prompt') or '',
            'state': state_padded.to(dtype=torch.bfloat16),
            'state_mask': state_mask,
            'action': action_padded.to(dtype=torch.bfloat16),
            'action_mask': action_mask,
            'embodiment_id': torch.tensor(self.arm_to_embodiment_id[arm_key], dtype=torch.long),
            'dataset_key': dataset_key,
        }
