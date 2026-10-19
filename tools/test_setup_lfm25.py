"""tools/test_setup_lfm25.py - Momoe: LFM2.5-24B-A2B and LFM2.5-8B-A1B on the ggml engine (GGML_FAMILIES["lfm25"],
["lfm25-8b"], momoe_setup): the size chosen by RAM, every offered size's file pinned with its size and SHA-256, each model
card's sampling, and the tokenizer's `lfm2` pre-tokenizer (llama.cpp's llama3 pattern, ignore_merges, the bos token
named in the pack)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_tokenizer  # noqa: E402

LFM = setup.GGML_FAMILIES["lfm25"]
LFM8 = setup.GGML_FAMILIES["lfm25-8b"]


class LfmSize(unittest.TestCase):
    def test_the_default_on_an_8_gb_jetson(self):
        self.assertEqual(setup.ggml_size(LFM, None, 7.3), "UD-IQ3_XXS")

    def test_the_default_with_more_ram(self):
        self.assertEqual(setup.ggml_size(LFM, None, 64), "UD-IQ3_XXS")

    def test_model_in_any_case(self):
        self.assertEqual(setup.ggml_size(LFM, "ud-q4_k_xl", 7.3), "UD-Q4_K_XL")


class Lfm8Size(unittest.TestCase):
    def test_the_default_on_an_8_gb_jetson(self):
        self.assertEqual(setup.ggml_size(LFM8, None, 7.3), "UD-Q4_K_XL")

    def test_q8_only_when_asked(self):
        self.assertEqual(setup.ggml_size(LFM8, None, 64), "UD-Q4_K_XL")
        self.assertEqual(setup.ggml_size(LFM8, "q8_0", 7.3), "Q8_0")


class LfmFiles(unittest.TestCase):
    def test_every_size_pinned(self):
        for fam, stem, least, repo in ((LFM, "LFM2.5-24B-A2B", 7 * 10 ** 9, "unsloth/LFM2.5-24B-A2B-GGUF"),
                                       (LFM8, "LFM2.5-8B-A1B", 3 * 10 ** 9, "unsloth/LFM2.5-8B-A1B-GGUF")):
            for q in fam["models"]:
                (path, name, size, sha), = setup.ggml_parts(fam, q)
                self.assertEqual(path, f"{stem}-{q}.gguf")
                self.assertEqual(path, name)
                self.assertGreater(size, least)
                self.assertEqual(len(sha), 64)
            self.assertIn(repo, setup.HF_REVISIONS)
            self.assertTrue(fam["hf"].endswith("/resolve/" + setup.HF_REVISIONS[repo] + "/"))
            self.assertIn(fam["default"], fam["models"])


class LfmSampling(unittest.TestCase):
    def test_the_cards(self):
        self.assertEqual(setup.recommended_sampling(LFM["name"] + "-ud-iq3_xxs"), {"temperature": 0.1, "top_k": 50})
        # the 8B's card says top_k 80; the server's sampler takes 1..64 (a config above it does not start)
        self.assertEqual(setup.recommended_sampling(LFM8["name"] + "-ud-q4_k_xl"), {"temperature": 0.2, "top_k": 64})


class LfmTokenizer(unittest.TestCase):
    def test_lfm2_is_llama3s_pattern(self):
        self.assertEqual(strata_tokenizer.PRE_PATTERNS["lfm2"], strata_tokenizer.CHATGLM4_PATTERN)
        self.assertIn("lfm2", strata_tokenizer.IGNORE_MERGES)
        self.assertIn("lfm2", strata_tokenizer.ADD_BOS)


if __name__ == "__main__":
    unittest.main()
