__all__ = [
    "EVO1",
    "Evo1ModelConfig",
    "LeRobotDataset",
    "build_dataloader",
    "build_dataset",
    "build_model",
    "evo1_collate_fn",
]


def __getattr__(name):
    if name in {"EVO1", "Evo1ModelConfig", "build_model"}:
        from .model import EVO1, Evo1ModelConfig, build_model

        exports = {
            "EVO1": EVO1,
            "Evo1ModelConfig": Evo1ModelConfig,
            "build_model": build_model,
        }
        return exports[name]
    if name in {"LeRobotDataset", "build_dataloader", "build_dataset", "evo1_collate_fn"}:
        from .dataset import LeRobotDataset, build_dataloader, build_dataset, evo1_collate_fn

        exports = {
            "LeRobotDataset": LeRobotDataset,
            "build_dataloader": build_dataloader,
            "build_dataset": build_dataset,
            "evo1_collate_fn": evo1_collate_fn,
        }
        return exports[name]
    raise AttributeError(name)
