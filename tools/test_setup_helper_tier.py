"""Tests for setup.py's helper tier (--helper-tier, docs/SECOND_GPU.md) and the server's handling of its config: no GPU,
no network, nothing installed.

    python tools/test_setup_helper_tier.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serve"))
import setup  # noqa: E402

TI = {"index": 1, "name": "NVIDIA GeForce RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}
G3060 = {"index": 0, "name": "NVIDIA GeForce RTX 3060", "vram_gb": 12.0, "arch": "86", "driver": "580", "unified": False}
TI2 = {**TI, "index": 0}


def args(**kw):
    base = {"gpus": None, "gpu": None, "helper_tier": False, "yes": True, "check": False}
    base.update(kw)
    return types.SimpleNamespace(**base)


def quiet(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        return fn(*a, **kw), out.getvalue()


class HelperPair(unittest.TestCase):
    def test_two_different_cards_are_a_pair_the_newer_first(self):
        can = setup.together_ok([G3060, TI])
        self.assertEqual([g["index"] for g in setup.helper_pair(can)], [1, 0])

    def test_two_equal_cards_are_not(self):
        self.assertEqual(setup.helper_pair(setup.together_ok([TI, {**TI2, "index": 2}])), [])


class ChooseGpus(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(setup, "gpu_free_gb", return_value=9.6)
        p.start()
        self.addCleanup(p.stop)

    def test_offered_for_two_different_cards_and_chosen(self):
        a = args(yes=False)
        with mock.patch("builtins.input", return_value="2"):
            sel, text = quiet(setup.choose_gpus, a, [G3060, TI])
        self.assertEqual(sel, [1, 0])
        self.assertTrue(a.helper_tier)
        self.assertIn("helper tier", text)
        self.assertIn("about 8 GB", text)          # 9.6 GB free less the 1.6 GB it keeps for its prompt buffers

    def test_the_split_stays_the_recommendation(self):
        a = args()
        sel, text = quiet(setup.choose_gpus, a, [G3060, TI])
        self.assertEqual(sel, [1, 0])
        self.assertFalse(a.helper_tier)
        self.assertIn("2) GPU 1 (NVIDIA GeForce RTX 4070 Ti, 12 GB) + GPU 0", text)

    def test_not_offered_for_two_equal_cards(self):
        a = args()
        _, text = quiet(setup.choose_gpus, a, [TI2, {**TI, "index": 1}])
        self.assertNotIn("helper tier", text)

    def test_the_option_picks_the_pair(self):
        a = args(helper_tier=True)
        sel, _ = quiet(setup.choose_gpus, a, [G3060, TI])
        self.assertEqual(sel, [1, 0])
        self.assertTrue(a.helper_tier)

    def test_the_option_on_equal_cards_falls_back(self):
        a = args(helper_tier=True)
        _, text = quiet(setup.choose_gpus, a, [TI2, {**TI, "index": 1}])
        self.assertFalse(a.helper_tier)
        self.assertIn("same card", text)

    def test_named_cards_need_two(self):
        with self.assertRaises(SystemExit):
            quiet(setup.choose_gpus, args(helper_tier=True, gpus="1"), [G3060, TI])


class Server(unittest.TestCase):
    def setUp(self):
        try:
            import server  # noqa: F401
        except ImportError as e:   # the server's own packages (the install's venv)
            self.skipTest(f"serve/server.py does not import here: {e}")
        self.server = sys.modules["server"]

    def test_no_layer_split_and_the_order_kept(self):
        cfg = {"gpu": [1, 0], "args": ["--helper-tier"]}
        self.assertFalse(self.server.layer_split_of(cfg))
        self.assertNotIn("--layer-split", self.server.engine_args(cfg))
        self.assertEqual(self.server.ordered_gpus(cfg, scores={1: 2.0, 0: 1.0}), [1, 0])

    def test_without_it_two_cards_are_a_split(self):
        self.assertTrue(self.server.layer_split_of({"gpu": [1, 0], "args": []}))


if __name__ == "__main__":
    unittest.main()
