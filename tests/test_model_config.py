import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "Evo_1"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from model.config import Evo1ModelConfig, normalize_model_config


class Evo1ModelConfigTest(unittest.TestCase):
    def test_normalize_action_dimension_defaults(self):
        config = normalize_model_config({"horizon": 8, "per_action_dim": 6})
        self.assertEqual(config["action_dim"], 48)
        self.assertEqual(config["action_horizon"], 8)

    def test_reject_inconsistent_action_dimension(self):
        with self.assertRaises(ValueError):
            Evo1ModelConfig.from_mapping({"horizon": 4, "per_action_dim": 3, "action_dim": 20})

    def test_unknown_action_head_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            Evo1ModelConfig.from_mapping({"action_head": "other"})


if __name__ == "__main__":
    unittest.main()
