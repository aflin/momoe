"""tools/test_setup_mimo25.py - Momoe: MiMo-V2.5 on the ggml engine (GGML_FAMILIES["mimo25"], momoe_setup): its size
pinned with each shard's size and SHA-256 at the pinned revision, the card's sampling, the family in the menu, the
config on a 12 GB card (2048-token prompt chunks, no draft vocabulary), and its `qwen2` pre-tokenizer."""
from __future__ import annotations

import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_tokenizer  # noqa: E402

FAM = setup.GGML_FAMILIES["mimo25"]


class Mimo25Family(unittest.TestCase):
    def test_the_size_pinned(self):
        self.assertTrue(FAM["hf"].endswith("/resolve/" + setup.HF_REVISIONS["unsloth/MiMo-V2.5-GGUF"] + "/"))
        parts = setup.ggml_parts(FAM, "UD-IQ2_XXS")
        self.assertEqual([p[1] for p in parts], ["MiMo-V2.5-UD-IQ2_XXS-0000%d-of-00003.gguf" % i for i in (1, 2, 3)])
        self.assertTrue(all(p[0] == "UD-IQ2_XXS/" + p[1] for p in parts))
        self.assertEqual(sum(p[2] for p in parts), 5953760 + 49525684032 + 46948527360)
        self.assertTrue(all(len(p[3]) == 64 for p in parts))

    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(FAM["name"] + "-ud-iq2_xxs"), {"temperature": 1.0, "top_p": 0.95})

    def test_in_the_menu(self):
        self.assertIn("mimo25", setup.MOMOE_MENU)

    def test_no_draft_vocabulary(self):
        self.assertFalse(FAM.get("own_mtp"))

    def run_setup(self, **extra):
        """momoe_setup up to the config on a 12 GB card (no build, no download): the config it would write."""
        written = {}
        a = mock.Mock(gpu=None, gpus=None, family="mimo25", model=None, context=None, host=None, api_key=None,
                      check=False, vision_tokens=None, gguf_dir=None, config_name=None, no_start=True, yes=True,
                      draft_vocab=None, models_dir="/m", helper_tier=False, layer_split=None, backend=None, vision=None,
                      kv=None)
        for k, v in extra.items():
            setattr(a, k, v)
        card = [{"index": 0, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]
        patches = [mock.patch.object(setup, "gpus", return_value=card), mock.patch.object(setup, "gpu_problem", return_value=None),
                   mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"),
                   mock.patch.object(setup, "ram_gb", return_value=63.0), mock.patch.object(setup, "say"),
                   mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), mock.patch.object(setup, "step"),
                   mock.patch.object(setup, "find_nvcc", return_value=("/nvcc", (13, 0))),
                   mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "run"),
                   mock.patch.object(setup, "stamp_ggml_build"),
                   mock.patch.object(setup.Path, "exists", return_value=True), mock.patch.object(setup, "done", return_value=True),
                   mock.patch.object(setup, "download", side_effect=AssertionError("downloaded")),
                   mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "check_shards"),
                   mock.patch.object(setup, "confirm_risk"),
                   mock.patch.object(setup, "name_config", side_effect=lambda p, a: p),
                   mock.patch.object(setup, "write_setup_config", side_effect=lambda p, c: written.update(c))]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            setup.momoe_setup(a, Path("/d"), 8080, "mimo25")
        return written

    def test_the_config(self):
        cfg = self.run_setup()
        a = cfg["args"]
        self.assertEqual(a[a.index("-m") + 1], "/m/mimo25-UD-IQ2_XXS/MiMo-V2.5-UD-IQ2_XXS-00001-of-00003.gguf")
        self.assertEqual(a[a.index("--ctx") + 1], "32768")
        self.assertEqual(a[a.index("--chunk") + 1], "2048")
        self.assertNotIn("--draft-vocab", a)
        self.assertNotIn("--mtp", a)                    # (its MTP blocks are not run)
        self.assertEqual(cfg["model_name"], "mimo-v2.5-ud-iq2_xxs")


class Mimo25Tokenizer(unittest.TestCase):
    """llama-vocab.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN2: qwen35's split without the combining marks (the pack gave llama.cpp's
    tokens for three prompts, 48-1,531 tokens)."""

    def test_the_split(self):
        import regex
        pat = regex.compile(strata_tokenizer.PRE_PATTERNS["qwen2"])
        self.assertEqual(pat.findall("Hello world's 12345 test!!\n\n  x"),
                         ["Hello", " world", "'s", " ", "1", "2", "3", "4", "5", " test", "!!\n\n", " ", " x"])
        # a combining mark is not part of the letters before it (qwen35 keeps it there)
        self.assertEqual(pat.findall("éx"), ["e", "́x"])
        self.assertEqual(regex.compile(strata_tokenizer.PRE_PATTERNS["qwen35"]).findall("éx"), ["éx"])


if __name__ == "__main__":
    unittest.main()
