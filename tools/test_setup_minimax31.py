"""tools/test_setup_minimax31.py - Momoe: MiniMax M3.1 Flash on the ggml engine (GGML_FAMILIES["minimax31"], momoe_setup):
its size pinned with each shard's size and SHA-256 at the pinned revision, the card's sampling, the family in the menu,
its license said, the config on two 12 GB cards (their layers split, the engine's settings for them, 512-token
chunks, a 16K context), the Mac's pool, and its `minimax-m2` pre-tokenizer."""
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

FAM = setup.GGML_FAMILIES["minimax31"]
CARDS = [{"index": 0, "name": "RTX 3060", "vram_gb": 12.0, "arch": "86", "driver": "580", "unified": False},
         {"index": 1, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]


class Minimax31Family(unittest.TestCase):
    def test_the_size_pinned(self):
        self.assertTrue(FAM["hf"].endswith("/resolve/" + setup.HF_REVISIONS["unsloth/MiniMax-M3.1-Flash-GGUF"] + "/"))
        parts = setup.ggml_parts(FAM, "UD-IQ2_XXS")
        self.assertEqual([p[1] for p in parts], ["MiniMax-M3.1-Flash-UD-IQ2_XXS-0000%d-of-00004.gguf" % i for i in (1, 2, 3, 4)])
        self.assertTrue(all(p[0] == "UD-IQ2_XXS/" + p[1] for p in parts))
        self.assertEqual(sum(p[2] for p in parts), 8243332 + 49926404000 + 49118488576 + 35894753536)
        self.assertTrue(all(len(p[3]) == 64 for p in parts))

    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(FAM["name"] + "-ud-iq2_xxs"), {"temperature": 1.0, "top_p": 0.95})

    def test_in_the_menu(self):
        self.assertIn("minimax31", setup.MOMOE_MENU)

    def test_the_license_says_the_commercial_terms(self):
        self.assertIn("Built with MiniMax M3.1-Flash", FAM["license"])
        self.assertIn("non-commercial", FAM["license"])

    def test_the_mac_pool_leaves_room(self):
        # its file as momoe_setup reads it (other weights 9.16 GiB, KV 122,880 bytes a token at 32K): 20 GiB on a 64 GB Mac
        pool = setup.mac_pool_gib(64, 116.51, 9.16, 0.0, 122880 * 32768 / 2**30) - FAM["mac_margin_gib"]
        self.assertEqual(pool, 20)

    def run_setup(self, cards, **extra):
        """momoe_setup up to the config (no build, no download): the config it would write, and what it said."""
        written, said = {}, []
        a = mock.Mock(gpu=None, gpus=None, family="minimax31", model=None, context=None, host=None, api_key=None,
                      check=False, vision_tokens=None, gguf_dir=None, config_name=None, no_start=True, yes=True,
                      draft_vocab=None, models_dir="/m", helper_tier=False, layer_split=None, backend=None, vision=None,
                      kv=None)
        for k, v in extra.items():
            setattr(a, k, v)
        patches = [mock.patch.object(setup, "gpus", return_value=cards), mock.patch.object(setup, "gpu_problem", return_value=None),
                   mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"),
                   mock.patch.object(setup, "ram_gb", return_value=63.0),
                   mock.patch.object(setup, "say", side_effect=lambda *x: said.append(" ".join(map(str, x)))),
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
            setup.momoe_setup(a, Path("/d"), 8080, "minimax31")
        return written, said

    def test_two_12_gb_cards_share_it(self):
        cfg, said = self.run_setup(CARDS)
        a = cfg["args"]
        self.assertEqual(a[a.index("-m") + 1], "/m/minimax31-UD-IQ2_XXS/MiniMax-M3.1-Flash-UD-IQ2_XXS-00001-of-00004.gguf")
        self.assertEqual(a[a.index("--chunk") + 1], "512")
        self.assertEqual(a[a.index("--ctx") + 1], "16384")      # (card_context: 32K's KV cache did not fit them)
        self.assertEqual(sorted(cfg["gpu"]), [0, 1])
        self.assertEqual(cfg["env"]["MOMOE_POOL_MIN_TOKENS"], "4")
        self.assertEqual(cfg["env"]["MOMOE_VRAM_MARGIN_MIB"], "1536")
        self.assertEqual(cfg["model_name"], "minimax-m3.1-flash-ud-iq2_xxs")
        self.assertTrue(any("Built with MiniMax M3.1-Flash" in line for line in said))   # (its license, before the download)


class Minimax31Tokenizer(unittest.TestCase):
    """llama-vocab.cpp's LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2 (the tokenizer.json original): case runs, contractions on the
    word, digits in threes (the pack gave llama.cpp's tokens for three prompts, 215-1,658 tokens, and one of 6,246)."""

    def test_the_split(self):
        import regex
        pat = regex.compile(strata_tokenizer.PRE_PATTERNS["minimax-m2"])
        self.assertEqual(pat.findall("Hello World's HTTPServer 12345 test!!\n\n  x/y"),
                         ["Hello", " World's", " HTTPServer", " ", "123", "45", " test", "!!\n\n", " ", " x", "/y"])


if __name__ == "__main__":
    unittest.main()
