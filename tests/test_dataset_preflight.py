import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Evo_1" / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from preflight import infer_dataset_dims, iter_dataset_videos, resolve_dataset_path


class DatasetPreflightTest(unittest.TestCase):
    def test_infer_dims_from_episodes_stats_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            meta_dir = dataset_root / "meta"
            meta_dir.mkdir(parents=True)
            payload = {
                "stats": {
                    "observation.state": {"min": [0.0, 1.0, 2.0], "max": [1.0, 2.0, 3.0]},
                    "action": {"min": [0.0, 1.0], "max": [1.0, 2.0]},
                }
            }
            (meta_dir / "episodes_stats.jsonl").write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )

            state_dim, action_dim = infer_dataset_dims(str(dataset_root))
            self.assertEqual(state_dim, 3)
            self.assertEqual(action_dim, 2)

    def test_iter_dataset_videos_returns_sorted_mp4s(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            videos_dir = dataset_root / "videos" / "chunk-000" / "cam"
            videos_dir.mkdir(parents=True)
            (videos_dir / "episode_0002.mp4").write_bytes(b"")
            (videos_dir / "episode_0001.mp4").write_bytes(b"")

            videos = iter_dataset_videos(dataset_root)
            self.assertEqual([path.name for path in videos], ["episode_0001.mp4", "episode_0002.mp4"])

    def test_resolve_dataset_path_rejects_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            resolve_dataset_path("/tmp/definitely_missing_evo1_dataset")


if __name__ == "__main__":
    unittest.main()
