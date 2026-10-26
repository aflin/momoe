"""tools/test_setup_ling3.py - Momoe: Ling-3.0-flash Sante on the ggml engine (GGML_FAMILIES["ling3sante"], momoe_setup):
its size pinned with each shard's size and SHA-256 at the pinned revision, the card's sampling, the family in the menu,
no draft vocabulary (the shipped subsets are Qwen's token ids), and its `bailingmoe2` pre-tokenizer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_tokenizer  # noqa: E402

FAM = setup.GGML_FAMILIES["ling3sante"]


class Ling3Family(unittest.TestCase):
    def test_the_size_pinned(self):
        repo = "bartowski/Ling-3.0-flash-Sante-GGUF"
        self.assertTrue(FAM["hf"].endswith("/resolve/" + setup.HF_REVISIONS[repo] + "/"))
        parts = setup.ggml_parts(FAM, "IQ3_XXS")
        self.assertEqual([p[1] for p in parts], ["Ling-3.0-flash-Sante-IQ3_XXS-0000%d-of-00002.gguf" % i for i in (1, 2)])
        self.assertTrue(all(p[0] == "Ling-3.0-flash-Sante-IQ3_XXS/" + p[1] for p in parts))
        self.assertEqual(sum(p[2] for p in parts), 39474010144 + 16195273472)
        self.assertTrue(all(len(p[3]) == 64 for p in parts))

    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(FAM["name"] + "-iq3_xxs"),
                         {"temperature": 0.6, "top_p": 0.95, "top_k": 20})

    def test_in_the_menu(self):
        self.assertIn("ling3sante", setup.MOMOE_MENU)

    def test_no_draft_vocabulary(self):
        self.assertFalse(FAM.get("own_mtp"))

    def run_setup(self, **extra):
        """momoe_setup up to the config on a 12 GB card (no build, no download): the config it would write."""
        written = {}
        a = mock.Mock(gpu=None, gpus=None, family="ling3sante", model=None, context=None, host=None, api_key=None,
                      check=False, vision_tokens=None, gguf_dir=None, config_name=None, no_start=True, yes=True,
                      draft_vocab=None, models_dir="/m", helper_tier=False, layer_split=None, backend=None, vision=None,
                      kv=None)
        for k, v in extra.items():
            setattr(a, k, v)
        card = [{"index": 0, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]
        with mock.patch.object(setup, "gpus", return_value=card), mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup, "ram_gb", return_value=63.0), mock.patch.object(setup, "say"), \
                mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), mock.patch.object(setup, "step"), \
                mock.patch.object(setup, "find_nvcc", return_value=("/nvcc", (13, 0))), \
                mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "run"), \
                mock.patch.object(setup, "stamp_ggml_build"), \
                mock.patch.object(setup.Path, "exists", return_value=True), mock.patch.object(setup, "done", return_value=True), \
                mock.patch.object(setup, "download", side_effect=AssertionError("downloaded")), \
                mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "check_shards"), \
                mock.patch.object(setup, "name_config", side_effect=lambda p, a: p), \
                mock.patch.object(setup, "write_setup_config", side_effect=lambda p, c: written.update(c)):
            setup.momoe_setup(a, Path("/d"), 8080, "ling3sante")
        return written

    def test_the_config(self):
        cfg = self.run_setup()
        a = cfg["args"]
        self.assertEqual(a[a.index("-m") + 1], "/m/ling3sante-IQ3_XXS/Ling-3.0-flash-Sante-IQ3_XXS-00001-of-00002.gguf")
        self.assertEqual(a[a.index("--ctx") + 1], "65536")
        self.assertNotIn("--draft-vocab", a)
        self.assertNotIn("--mtp", a)                    # (on a 12 GB card drafting measured slower: 20.8-21.5 vs 22.5)
        self.assertEqual(cfg["model_name"], "ling-3.0-flash-sante-iq3_xxs")

    def test_gguf_dir_is_absolute(self):
        cfg = self.run_setup(gguf_dir="models/ling")
        m = cfg["args"][cfg["args"].index("-m") + 1]
        self.assertTrue(Path(m).is_absolute(), m)


class Ling3Tokenizer(unittest.TestCase):
    """llama-vocab.cpp's LLAMA_VOCAB_PRE_TYPE_BAILINGMOE: words with their leading space, one digit a piece, punctuation
    runs with the line breaks after them (the pack gave llama.cpp's tokens for three prompts, 44-1,589 tokens)."""

    def test_the_split(self):
        import regex
        pat = regex.compile(strata_tokenizer.PRE_PATTERNS["bailingmoe2"])
        self.assertEqual(pat.findall("Hello world's 12345 test!!\n\n  x"),
                         ["Hello", " world", "'s", " ", "1", "2", "3", "4", "5", " test", "!!\n\n", " ", " x"])
        self.assertIs(strata_tokenizer.PRE_PATTERNS["bailingmoe"], strata_tokenizer.PRE_PATTERNS["bailingmoe2"])


if __name__ == "__main__":
    unittest.main()
