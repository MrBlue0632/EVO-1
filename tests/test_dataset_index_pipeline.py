import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "Evo_1"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from dataset.build import _resolve_manifest_path
    from dataset.cached_dataset import CachedLeRobotDataset
    from dataset.exporter import build_export_key
except ModuleNotFoundError:
    _resolve_manifest_path = None
    CachedLeRobotDataset = None
    build_export_key = None


@unittest.skipIf(_resolve_manifest_path is None, "dataset modules are not available in current environment")
class DatasetIndexPipelineTest(unittest.TestCase):
    def test_export_key_consistent_between_build_and_exporter(self):
        config = {
            "cache_dir": "/tmp/evo1_cache",
            "image_size": 448,
            "horizon": 16,
            "max_samples_per_file": None,
        }
        dataset_config = {"data_groups": {"arm_a": {"set_1": {"path": "/data/demo"}}}}
        manifest_path = _resolve_manifest_path(config, dataset_config)
        export_key = build_export_key(config, dataset_config)
        self.assertIn(f"/{export_key}/manifest.json", str(manifest_path))

    def test_cached_dataset_requires_schema_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            payload = {
                "export_key": "demo",
                "samples": [{"arm_key": "arm_a", "dataset_key": "set_1", "state": [0.0], "action": [[0.0]], "image_paths": []}],
                "arm_to_embodiment_id": {"arm_a": 0},
                "arm2stats_dict": {
                    "arm_a": {
                        "observation.state": {"min": [0.0], "max": [1.0]},
                        "action": {"min": [0.0], "max": [1.0]},
                    }
                },
                "max_action_dim": 1,
                "max_state_dim": 1,
                "max_views": 1,
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                CachedLeRobotDataset(manifest_path)


if __name__ == "__main__":
    unittest.main()
