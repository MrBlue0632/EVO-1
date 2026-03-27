from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, MutableMapping


@dataclass(frozen=True)
class Evo1ModelConfig:
    device: str = 'cuda'
    return_cls_only: bool = False
    vlm_name: str = 'OpenGVLab/InternVL3-1B'
    action_head: str = 'flowmatching'
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    state_dim: int = 7
    per_action_dim: int = 7
    horizon: int = 16
    action_dim: int = 112
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.0
    num_inference_timesteps: int = 50
    num_categories: int = 1
    finetune_vlm: bool = False
    finetune_action_head: bool = False

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> 'Evo1ModelConfig':
        config = dict(raw_config)
        horizon = int(config.get('action_horizon', config.get('horizon', cls.horizon)))
        per_action_dim = int(config.get('per_action_dim', cls.per_action_dim))
        action_dim = int(config.get('action_dim', horizon * per_action_dim))
        if action_dim != horizon * per_action_dim:
            raise ValueError(
                f'action_dim ({action_dim}) must equal horizon ({horizon}) * per_action_dim ({per_action_dim})'
            )

        action_head = str(config.get('action_head', cls.action_head)).lower()
        if action_head != 'flowmatching':
            raise NotImplementedError(f'Unknown action_head: {action_head}')

        return cls(
            device=str(config.get('device', cls.device)),
            return_cls_only=bool(config.get('return_cls_only', cls.return_cls_only)),
            vlm_name=str(config.get('vlm_name', cls.vlm_name)),
            action_head=action_head,
            embed_dim=int(config.get('embed_dim', cls.embed_dim)),
            hidden_dim=int(config.get('hidden_dim', cls.hidden_dim)),
            state_hidden_dim=int(config.get('state_hidden_dim', cls.state_hidden_dim)),
            state_dim=int(config.get('state_dim', cls.state_dim)),
            per_action_dim=per_action_dim,
            horizon=horizon,
            action_dim=action_dim,
            num_heads=int(config.get('num_heads', cls.num_heads)),
            num_layers=int(config.get('num_layers', cls.num_layers)),
            dropout=float(config.get('dropout', cls.dropout)),
            num_inference_timesteps=int(config.get('num_inference_timesteps', cls.num_inference_timesteps)),
            num_categories=int(config.get('num_categories', cls.num_categories)),
            finetune_vlm=bool(config.get('finetune_vlm', cls.finetune_vlm)),
            finetune_action_head=bool(config.get('finetune_action_head', cls.finetune_action_head)),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action_horizon'] = self.horizon
        return data


def normalize_model_config(raw_config: Mapping[str, Any]) -> Dict[str, Any]:
    return Evo1ModelConfig.from_mapping(raw_config).to_dict()


def merge_normalized_model_config(target: MutableMapping[str, Any], raw_config: Mapping[str, Any]) -> MutableMapping[str, Any]:
    normalized = normalize_model_config(raw_config)
    target.update(normalized)
    return target
