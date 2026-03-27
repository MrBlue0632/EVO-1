import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def dataset_cache_root(cache_dir=None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    env_cache = os.environ.get('EVO1_CACHE_DIR')
    if env_cache:
        return Path(env_cache).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / '.cache' / 'evo1'


def fingerprint_payload(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha1(encoded).hexdigest()[:16]


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_bytes(content)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True).encode('utf-8')
    atomic_write_bytes(path, content)


def normalize_minmax(value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    normalized = 2 * (value - min_value) / (max_value - min_value + 1e-8) - 1
    return torch.clamp(normalized, -1.0, 1.0)


def pad_tensor(source_tensor: torch.Tensor, max_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    source_dim = source_tensor.shape[-1]
    if source_dim > max_dim:
        raise ValueError(f'source dimension {source_dim} exceeds max_dim {max_dim}')

    if source_tensor.dim() > 1:
        padded_shape = (*source_tensor.shape[:-1], max_dim)
    else:
        padded_shape = (max_dim,)

    padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
    mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)
    data_slice = (..., slice(0, source_dim))
    padded_tensor[data_slice] = source_tensor
    mask[data_slice] = True
    return padded_tensor, mask


def collate_batch(batch):
    prompts = [item['prompt'] for item in batch]
    images = [item['images'] for item in batch]
    states = torch.stack([item['state'] for item in batch], dim=0)
    actions = torch.stack([item['action'] for item in batch], dim=0)
    action_mask = torch.stack([item['action_mask'] for item in batch], dim=0)
    image_masks = torch.stack([item['image_mask'] for item in batch], dim=0)
    state_mask = torch.stack([item['state_mask'] for item in batch], dim=0)
    embodiment_ids = torch.stack([item['embodiment_id'] for item in batch], dim=0)
    return {
        'prompts': prompts,
        'images': images,
        'states': states,
        'actions': actions,
        'action_mask': action_mask,
        'state_mask': state_mask,
        'image_masks': image_masks,
        'embodiment_ids': embodiment_ids,
    }
