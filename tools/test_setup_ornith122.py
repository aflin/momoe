"""tools/test_setup_ornith122.py - Momoe: Ornith-1.5-122B-A10B on the ggml engine (GGML_FAMILIES["ornith122"],
momoe_setup): the size, its file pinned with its size and SHA-256 at the pinned revision, Ornith's sampling, and the
family in the menu."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

FAM = setup.GGML_FAMILIES["ornith122"]


class Ornith122(unittest.TestCase):
    def test_the_size(self):
        for ram in (32, 64, 128):
            self.assertEqual(setup.ggml_size(FAM, None, ram), "UD-IQ3_XXS")

    def test_the_file_pinned(self):
        (path, name, size, sha), = setup.ggml_parts(FAM, "UD-IQ3_XXS")
        self.assertEqual(path, "Ornith-1.5-122B-UD-IQ3_XXS.gguf")
        self.assertEqual(path, name)
        self.assertEqual(size, 49515667712)
        self.assertEqual(len(sha), 64)
        repo = "unsloth/Ornith-1.5-122B-GGUF"
        self.assertTrue(FAM["hf"].endswith("/resolve/" + setup.HF_REVISIONS[repo] + "/"))

    def test_ornith_sampling(self):
        self.assertEqual(setup.recommended_sampling(FAM["name"] + "-ud-iq3_xxs"), setup.SAMPLING_QWEN36)

    def test_in_the_menu(self):
        self.assertIn("ornith122", setup.MOMOE_MENU)


if __name__ == "__main__":
    unittest.main()
