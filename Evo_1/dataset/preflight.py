import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def resolve_dataset_path(dataset_dir: str) -> Path:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def iter_dataset_videos(dataset_path: Path) -> List[Path]:
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_root}")
    return sorted(videos_root.rglob("*.mp4"))


def _read_first_stats_object(dataset_path: Path) -> Dict:
    stats_json = dataset_path / "meta" / "stats.json"
    if stats_json.exists():
        with open(stats_json, "r", encoding="utf-8") as handle:
            return json.load(handle)

    stats_jsonl = dataset_path / "meta" / "episodes_stats.jsonl"
    if stats_jsonl.exists():
        with open(stats_jsonl, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            raise ValueError(f"Stats file is empty: {stats_jsonl}")
        payload = json.loads(first_line)
        return payload.get("stats", payload)

    raise FileNotFoundError(
        f"Normalization stats file not found under {dataset_path / 'meta'}"
    )


def infer_dataset_dims(dataset_dir: str) -> Tuple[int, int]:
    dataset_path = resolve_dataset_path(dataset_dir)
    stats = _read_first_stats_object(dataset_path)
    try:
        state_dim = len(stats["observation.state"]["min"])
        action_dim = len(stats["action"]["min"])
    except KeyError as exc:
        raise KeyError(f"Missing stats key in dataset metadata: {exc}") from exc
    return state_dim, action_dim


def validate_video_files(
    video_paths: Iterable[Path],
    fail_fast: bool = False,
    max_invalid: Optional[int] = None,
) -> List[Dict[str, str]]:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyAV is required for video validation. Install the training environment and retry."
        ) from exc

    invalid_files: List[Dict[str, str]] = []
    for path in video_paths:
        try:
            with av.open(str(path)) as container:
                iterator = container.decode(video=0)
                next(iterator, None)
        except Exception as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})
            if fail_fast:
                break
            if max_invalid is not None and len(invalid_files) >= max_invalid:
                break
    return invalid_files
