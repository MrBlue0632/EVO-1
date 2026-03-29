#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = CURRENT_DIR.parent / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from preflight import infer_dataset_dims, iter_dataset_videos, resolve_dataset_path, validate_video_files


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LeRobot v2.1 dataset videos before EVO-1 training.")
    parser.add_argument("--dataset-dir", required=True, help="Root directory of the LeRobot dataset.")
    parser.add_argument("--report-path", default=None, help="Optional JSON report output path.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop scanning on the first invalid video.")
    parser.add_argument(
        "--max-invalid",
        type=int,
        default=None,
        help="Stop after recording this many invalid files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_dir)
    state_dim, action_dim = infer_dataset_dims(str(dataset_path))
    videos = iter_dataset_videos(dataset_path)
    if not videos:
        raise ValueError(f"No mp4 files found under {dataset_path / 'videos'}")

    invalid_files = validate_video_files(
        videos,
        fail_fast=args.fail_fast,
        max_invalid=args.max_invalid,
    )
    report = {
        "dataset_dir": str(dataset_path),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_dim": state_dim,
        "per_action_dim": action_dim,
        "video_count": len(videos),
        "invalid_count": len(invalid_files),
        "invalid_files": invalid_files,
    }

    if args.report_path:
        report_path = Path(args.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)

    print(f"dataset_dir={dataset_path}")
    print(f"state_dim={state_dim} per_action_dim={action_dim}")
    print(f"video_count={len(videos)} invalid_count={len(invalid_files)}")
    if invalid_files:
        print("Invalid videos detected:", file=sys.stderr)
        for item in invalid_files[:20]:
            print(f"{item['path']}: {item['error']}", file=sys.stderr)
        if len(invalid_files) > 20:
            print(f"... and {len(invalid_files) - 20} more", file=sys.stderr)
        return 1

    print("All videos passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
