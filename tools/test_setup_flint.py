"""tools/test_setup_flint.py - Momoe: Flint 40B-A2.5B on the ggml engine (GGML_FAMILIES["flint"], momoe_setup): the size
by machine (flint_size), both sizes pinned with their size and SHA-256 at the pinned revision, the card's sampling, the
family in the menu, its KV cache counted as its full-attention blocks per position and the sliding-window blocks' rings,
and its `flint` pre-tokenizer (three patterns one after another)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

FAM = setup.GGML_FAMILIES["flint"]


class FlintSize(unittest.TestCase):
    def test_by_machine(self):
        self.assertEqual(setup.flint_size(63, False, 11.9), "UD-IQ3_XXS")    # the RTX 4070 Ti: 90 against 67 tok/s
        self.assertEqual(setup.flint_size(32, True, 0), "UD-IQ3_XXS")       # the M4 Air: 42.2 against 16.6-19.1
        self.assertEqual(setup.flint_size(64, True, 0), "UD-Q4_K_XL")       # a 64 GB Mac: every expert in memory
        self.assertEqual(setup.flint_size(122, True, 0), "UD-Q4_K_XL")      # the DGX Spark
        self.assertEqual(setup.flint_size(64, False, 24), "UD-Q4_K_XL")     # a 24 GB card

    def test_both_sizes_pinned(self):
        repo = "unsloth/flint-GGUF"
        self.assertTrue(FAM["hf"].endswith("/resolve/" + setup.HF_REVISIONS[repo] + "/"))
        for q, size in (("UD-IQ3_XXS", 14535427520), ("UD-Q4_K_XL", 25054688704)):
            (path, name, n, sha), = setup.ggml_parts(FAM, q)
            self.assertEqual(path, f"flint-{q}.gguf")
            self.assertEqual((n, len(sha)), (size, 64))

    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(FAM["name"] + "-ud-iq3_xxs"),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 20})

    def test_in_the_menu(self):
        self.assertIn("flint", setup.MOMOE_MENU)


class FlintKv(unittest.TestCase):
    MD = {"general.architecture": "flint", "flint.block_count": 49, "flint.attention.head_count_kv": 4,
          "flint.attention.key_length": 128, "flint.attention.head_count": 32, "flint.embedding_length": 2048,
          # the file's own pattern (UD-IQ3_XXS): 14 full-attention blocks, 35 sliding-window ones
          "flint.attention.sliding_window_pattern": [c == "T" for c in "FFTTTFTTTFTTTFTTTFTTTFTTTFTTTFTTTFTTTFTTTFTTTFTTF"]}

    def files(self):
        g = mock.Mock(metadata=self.MD)
        return mock.patch("gguf_reader.GGUFFile", return_value=g)

    def test_full_blocks_per_position_rings_fixed(self):
        import gguf_reader  # noqa: F401
        with self.files():
            # 14 full-attention blocks keep every position: 14 * 4 heads * 128 * 2 (K, V) * 2 bytes
            self.assertEqual(setup.kv_bytes_per_token([Path("x.gguf")]), 14 * 4 * 128 * 2 * 2)
            # 35 sliding-window blocks: a ring of KV_RING_ROWS positions each
            self.assertAlmostEqual(setup.kv_ring_gib([Path("x.gguf")]), 35 * setup.KV_RING_ROWS * 4 * 128 * 4 / 2**30)


class FlintTokenizer(unittest.TestCase):
    def test_three_patterns_in_a_row(self):
        import strata_tokenizer as ST
        self.assertEqual(len(ST.PRE_SEQUENCES["flint"]), 3)
        tk = ST.Tokenizer.__new__(ST.Tokenizer)
        import regex
        tk._seq = [regex.compile(x) for x in ST.PRE_SEQUENCES["flint"]]
        # digits in threes first, CJK runs apart, then words with their contractions (the same split llama.cpp makes:
        # the tokens of 14 texts matched its /tokenize, checked against llama.cpp a37cc46)
        self.assertEqual(tk._split_seq("abc12345 日本語 don't"), ["abc", "123", "45", " ", "日本語", " don't"])

if __name__ == "__main__":
    unittest.main()
