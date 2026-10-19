"""tools/test_setup_bolt.py - Momoe: Nemotron 3.5 Bolt on the ggml engine (GGML_FAMILIES, momoe_setup): the size chosen by
RAM, --model in any case, the model card's sampling, and every offered size pinned with its size and SHA-256."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

BOLT = setup.GGML_FAMILIES["bolt"]


class BoltSize(unittest.TestCase):
    def test_the_measured_size_on_63_gb(self):
        self.assertEqual(setup.ggml_size(BOLT, None, 62.7), "UD-IQ3_XXS")

    def test_the_larger_size_with_more_ram(self):
        self.assertEqual(setup.ggml_size(BOLT, None, 96), "UD-Q3_K_XL")

    def test_the_smallest_when_none_fits(self):
        self.assertEqual(setup.ggml_size(BOLT, None, 32), "UD-IQ3_XXS")

    def test_model_in_any_case(self):
        self.assertEqual(setup.ggml_size(BOLT, "ud-q3_k_xl", 32), "UD-Q3_K_XL")

    def test_model_it_does_not_have(self):
        with self.assertRaises(SystemExit):
            setup.ggml_size(BOLT, "UD-Q4_K_XL", 64)


class BoltFiles(unittest.TestCase):
    def test_every_size_pinned(self):
        for q in BOLT["models"]:
            size, sha = BOLT["sha256"][BOLT["file"].format(q=q)]
            self.assertGreater(size, 10 ** 10)
            self.assertEqual(len(sha), 64)
        self.assertIn("unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF", setup.HF_REVISIONS)

    def test_family_and_sizes_accepted(self):
        self.assertIn("bolt", setup.GGML_FAMILIES)
        for q in BOLT["models"]:
            self.assertIn(q, setup.model_choices())


class BoltSampling(unittest.TestCase):
    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(f"{BOLT['name']}-ud-iq3_xxs"), {"temperature": 1.0, "top_p": 0.95})


if __name__ == "__main__":
    unittest.main()
