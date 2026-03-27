import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "Evo_1"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import torch
    from dataset.utils import collate_batch, normalize_minmax, pad_tensor
except ModuleNotFoundError:
    torch = None
    collate_batch = None
    normalize_minmax = None
    pad_tensor = None


@unittest.skipIf(torch is None, "torch is not installed in the current environment")
class DatasetUtilsTest(unittest.TestCase):
    def test_pad_tensor_returns_mask(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        padded, mask = pad_tensor(source, 4)
        self.assertEqual(tuple(padded.shape), (2, 4))
        self.assertTrue(torch.equal(mask[:, :2], torch.ones(2, 2, dtype=torch.bool)))
        self.assertTrue(torch.equal(mask[:, 2:], torch.zeros(2, 2, dtype=torch.bool)))

    def test_normalize_minmax_clamps(self):
        value = torch.tensor([0.0, 5.0, 10.0])
        normalized = normalize_minmax(value, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
        self.assertTrue(torch.allclose(normalized, torch.tensor([-1.0, 0.0, 1.0])))

    def test_collate_batch_keeps_expected_keys(self):
        sample = {
            "prompt": "demo",
            "images": torch.zeros(3, 3, 4, 4),
            "state": torch.zeros(4),
            "action": torch.zeros(2, 4),
            "action_mask": torch.ones(2, 4, dtype=torch.bool),
            "image_mask": torch.ones(3, dtype=torch.bool),
            "state_mask": torch.ones(4, dtype=torch.bool),
            "embodiment_id": torch.tensor(1, dtype=torch.long),
        }
        batch = collate_batch([sample, sample])
        self.assertEqual(sorted(batch.keys()), [
            "action_mask",
            "actions",
            "embodiment_ids",
            "image_masks",
            "images",
            "prompts",
            "state_mask",
            "states",
        ])
        self.assertEqual(batch["states"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
