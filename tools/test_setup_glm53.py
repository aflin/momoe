"""tools/test_setup_glm53.py - Momoe: GLM-5.3-Flash on the ggml engine (GGML_FAMILIES["glm53"], momoe_setup): the size
chosen by RAM, every offered size's files pinned with their sizes and SHA-256, the model card's sampling, and a GPU that
shares the system's memory (the DGX Spark's GB10: nvidia-smi gives its memory as [N/A]) found as one."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

GLM = setup.GGML_FAMILIES["glm53"]


class GlmSize(unittest.TestCase):
    def test_the_default_on_the_spark(self):
        self.assertEqual(setup.ggml_size(GLM, None, 121.7), "UD-IQ3_XXS")

    def test_the_smallest_on_a_64_gb_mac(self):
        # no size fits 64 GB: the smallest file (UD-IQ3_XXS, 121 GB), not the first listed (UD-IQ4_XS, 157 GB)
        self.assertEqual(setup.ggml_size(GLM, None, 64), "UD-IQ3_XXS")

    def test_the_larger_size_when_asked(self):
        self.assertEqual(setup.ggml_size(GLM, "UD-IQ4_XS", 121.7), "UD-IQ4_XS")

    def test_model_in_any_case(self):
        self.assertEqual(setup.ggml_size(GLM, "ud-iq3_xxs", 121.7), "UD-IQ3_XXS")


class GlmFiles(unittest.TestCase):
    def test_every_size_pinned(self):
        for q in GLM["models"]:
            parts = setup.ggml_parts(GLM, q)
            self.assertGreater(len(parts), 1)
            self.assertTrue(parts[0][1].endswith("-00001-of-%05d.gguf" % len(parts)))
            for path, name, size, sha in parts:
                self.assertTrue(path.startswith(q + "/") and path.endswith(name))
                self.assertGreater(size, 10 ** 6)
                self.assertEqual(len(sha), 64)
        self.assertIn("unsloth/GLM-5.3-Flash-GGUF", setup.HF_REVISIONS)
        self.assertIn(GLM["hf"].split("/resolve/")[1].rstrip("/"), setup.HF_REVISIONS.values())

    def test_one_file_families_unchanged(self):
        bolt = setup.GGML_FAMILIES["bolt"]
        name = bolt["file"].format(q="UD-IQ3_XXS")
        self.assertEqual(setup.ggml_parts(bolt, "UD-IQ3_XXS"), [(name, name, *bolt["sha256"][name])])

    def test_sizes_accepted(self):
        for q in GLM["models"]:
            self.assertIn(q, setup.model_choices())


class GlmSampling(unittest.TestCase):
    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling("glm-5.3-flash-ud-iq4_xs"), {"temperature": 1.0, "top_p": 0.95})


class UnifiedGpu(unittest.TestCase):
    def test_gb10_found_with_the_ram_as_its_memory(self):
        smi = "0, NVIDIA GB10, [N/A], 12.1, 580.159.03\n"
        with mock.patch.object(setup, "out", return_value=smi), mock.patch.object(setup, "ram_gb", return_value=121.7):
            g = setup.gpus()
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0]["unified"])
        self.assertAlmostEqual(g[0]["vram_gb"], 121.7)
        self.assertEqual(g[0]["arch"], "121")

    def test_a_card_with_its_own_memory(self):
        smi = "1, NVIDIA GeForce RTX 4070 Ti, 12282, 8.9, 580.95.05\n"
        with mock.patch.object(setup, "out", return_value=smi):
            g = setup.gpus()
        self.assertFalse(g[0]["unified"])
        self.assertAlmostEqual(g[0]["vram_gb"], 12282 / 1024)


if __name__ == "__main__":
    unittest.main()
