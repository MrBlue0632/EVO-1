from .config import Evo1ModelConfig, merge_normalized_model_config, normalize_model_config

__all__ = [
    "EVO1",
    "Evo1ModelConfig",
    "build_model",
    "merge_normalized_model_config",
    "normalize_model_config",
]


def __getattr__(name):
    if name in {"EVO1", "build_model"}:
        from .evo1_model import EVO1, build_model

        exports = {"EVO1": EVO1, "build_model": build_model}
        return exports[name]
    raise AttributeError(name)
