"""tools/test_setup_gemma4_lightning.py - Momoe: Gemma 4 26B-A4B and Nemotron 3.5 Lightning 30B-A3B on the ggml engine
(GGML_FAMILIES["gemma4"], ["lightning"], momoe_setup): the size by machine (gemma4_size, lightning_size), both sizes of
each pinned with their size and SHA-256 at the pinned revision, the cards' sampling, the families in the menu, Gemma 4's
KV cache counted as its full-attention blocks' per position (their own KV heads and head size) and the sliding-window
blocks' rings, and its tokenizer (SentencePiece-style BPE: spaces as U+2581, byte tokens)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

GEMMA = setup.GGML_FAMILIES["gemma4"]
LIGHT = setup.GGML_FAMILIES["lightning"]


class Sizes(unittest.TestCase):
    def test_gemma4_by_machine(self):
        self.assertEqual(setup.gemma4_size(63, False, 11.9), "UD-IQ3_XXS")   # the RTX 4070 Ti: 92 against 75 tok/s
        self.assertEqual(setup.gemma4_size(64, False, 24), "UD-Q4_K_XL")     # a 24 GB card holds it
        self.assertEqual(setup.gemma4_size(32, True, 0), "UD-IQ3_XXS")       # a 32 GB Mac
        self.assertEqual(setup.gemma4_size(64, True, 0), "UD-Q4_K_XL")       # a 64 GB Mac
        self.assertEqual(setup.gemma4_size(122, True, 0), "UD-Q4_K_XL")      # the DGX Spark

    def test_lightning_by_machine(self):
        self.assertEqual(setup.lightning_size(63, False, 11.9), "UD-IQ3_XXS")   # the RTX 4070 Ti: 71 against 38 tok/s
        self.assertEqual(setup.lightning_size(64, False, 24), "UD-Q4_K_XL")
        self.assertEqual(setup.lightning_size(32, True, 0), "UD-IQ3_XXS")
        self.assertEqual(setup.lightning_size(64, True, 0), "UD-Q4_K_XL")
        self.assertEqual(setup.lightning_size(122, True, 0), "UD-Q4_K_XL")

    def test_rules_named(self):
        for fam in ("flint", "gemma4", "lightning"):
            self.assertTrue(callable(getattr(setup, setup.SIZE_RULES[fam])))

    def test_pinned(self):
        for fam, repo, stem, sizes in (
                (GEMMA, "unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-",
                 {"UD-IQ3_XXS": 11416548832, "UD-Q4_K_XL": 17010980576}),
                (LIGHT, "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-",
                 {"UD-IQ3_XXS": 19758307392, "UD-Q4_K_XL": 25505724480})):
            self.assertTrue(fam["hf"].endswith("/resolve/" + setup.HF_REVISIONS[repo] + "/"))
            for q, size in sizes.items():
                (path, name, n, sha), = setup.ggml_parts(fam, q)
                self.assertEqual(path, f"{stem}{q}.gguf")
                self.assertEqual((n, len(sha)), (size, 64))

    def test_the_cards_sampling(self):
        self.assertEqual(setup.recommended_sampling(GEMMA["name"] + "-ud-iq3_xxs"),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 64})
        self.assertEqual(setup.recommended_sampling(LIGHT["name"] + "-ud-iq3_xxs"), {"temperature": 1.0, "top_p": 0.95})

    def test_in_the_menu(self):
        self.assertIn("gemma4", setup.MOMOE_MENU)
        self.assertIn("lightning", setup.MOMOE_MENU)


class Gemma4Kv(unittest.TestCase):
    PAT = [c == "T" for c in "TTTTTF" * 5]          # the file's: 25 sliding-window blocks, 5 full
    MD = {"general.architecture": "gemma4", "gemma4.block_count": 30, "gemma4.attention.head_count": 16,
          "gemma4.attention.head_count_kv": [2 if not t else 8 for t in PAT], "gemma4.attention.key_length": 512,
          "gemma4.attention.key_length_swa": 256, "gemma4.embedding_length": 2816,
          "gemma4.attention.sliding_window_pattern": PAT}

    def test_full_blocks_per_position_rings_fixed(self):
        import gguf_reader  # noqa: F401
        with mock.patch("gguf_reader.GGUFFile", return_value=mock.Mock(metadata=self.MD)):
            # 5 full blocks: 2 KV heads * 512 * 2 (K, V) * 2 bytes each
            self.assertEqual(setup.kv_bytes_per_token([Path("x.gguf")]), 5 * 2 * 512 * 2 * 2)
            # 25 sliding-window blocks: a ring of KV_RING_ROWS positions, 8 KV heads * 256
            self.assertAlmostEqual(setup.kv_ring_gib([Path("x.gguf")]), 25 * setup.KV_RING_ROWS * 8 * 256 * 4 / 2**30)


class Gemma4Tokenizer(unittest.TestCase):
    def tok(self):
        import strata_tokenizer as ST
        tokens = ["<pad>", "<eos>", "<bos>", "\n", "\n\n", "▁", "a", "b", "▁a", "ab", "<0xE2>", "<0x82>",
                  "<0xAC>"]
        types = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6]
        return ST.Tokenizer(tokens, ["▁ a", "a b"], types, pre="gemma4")

    def test_spaces_merges_bytes(self):
        tk = self.tok()
        # " ab": "▁" + "ab" (the merge a b ranks... after ▁ a: rank 0 wins, then b alone)
        self.assertEqual(tk.encode(" ab"), [8, 7])
        self.assertEqual(tk.encode("\n\n"), [4])                 # a run of line breaks that is one token
        self.assertEqual(tk.encode("€"), [10, 11, 12])      # outside the vocabulary: its UTF-8 bytes
        self.assertEqual(tk.decode([8, 7, 4, 10, 11, 12]), " ab\n\n€")

    def test_bos_written(self):
        import strata_tokenizer as ST
        self.assertIn("gemma4", ST.ADD_BOS)


if __name__ == "__main__":
    unittest.main()
