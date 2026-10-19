"""tools/test_setup_anywhere.py - Momoe: every GGUF family through Momoe's ggml engine (momoe_setup): momoe-metal on a
Mac, momoe-cuda on Linux; the Qwen families there too on a GPU that shares the RAM (the Spark, a Jetson).  The files
and sizes each family gets, the menu's choices, NVFP4 refused (Strata's own pack format), the routing, and a Mac's
expert pool sized to the model.

    python tools/test_setup_anywhere.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Files(unittest.TestCase):
    def test_flash_next_shards(self):
        files = setup.momoe_files("qwen", "IQ3_XXS")
        self.assertEqual([f[1] for f in files], ["Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                                                 "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"])
        self.assertTrue(files[0][0].endswith("/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"))

    def test_ornith_pinned(self):
        (url, name, size, sha), = setup.momoe_files("ornith", setup.MAC_MODEL)
        self.assertEqual(name, "Ornith-1.5-35B-A3B-IQ4_XS.gguf")
        self.assertEqual((size, sha), setup.ORNITH_FILES[name])

    def test_ggml_family_as_before(self):
        (url, name, size, sha), = setup.momoe_files("lfm25-8b", "UD-Q4_K_XL")
        self.assertEqual(url, setup.GGML_FAMILIES["lfm25-8b"]["hf"] + name)
        self.assertEqual((size, sha), setup.LFM25_8B_FILES[name])

    def test_every_menu_family_has_sizes_and_files(self):
        for f in setup.MOMOE_MENU:
            sizes = setup.momoe_sizes(f)
            self.assertTrue(sizes, f)
            for q in sizes:
                self.assertTrue(setup.momoe_files(f, q), (f, q))


class Sizes(unittest.TestCase):
    def test_given_size_any_case(self):
        self.assertEqual(setup.momoe_size("qwen", "iq3_xxs", 64), "IQ3_XXS")
        self.assertEqual(setup.momoe_size("lfm25", "ud-q4_k_xl", 8), "UD-Q4_K_XL")

    def test_unknown_size(self):
        with self.assertRaises(SystemExit):
            setup.momoe_size("qwen", "UD-Q9_X", 64)

    def test_main_paths_recommendation(self):
        names = list(setup.momoe_sizes("qwen"))
        self.assertEqual(setup.momoe_size("qwen", None, 128), setup.recommended_size("qwen", names, 128))
        self.assertEqual(setup.momoe_size("ornith", None, 32), setup.small_size("ornith", 32))

    def test_nvfp4_not_offered(self):
        self.assertNotIn("nvfp4", setup.MOMOE_MENU)


class Menu(unittest.TestCase):
    def ask_first(self, ram):
        a = mock.Mock(family=None, yes=True)
        with mock.patch.object(setup, "say"):
            return setup.momoe_menu(a, ram)

    def test_ornith_where_it_fits(self):
        self.assertEqual(self.ask_first(64), "ornith")

    def test_lfm_on_a_small_machine(self):
        self.assertEqual(self.ask_first(8), "lfm25")              # (the 24B's UD-IQ3_XXS: measured on 8 GB)

    def test_family_not_here(self):
        a = mock.Mock(family="nvfp4", yes=True)
        with self.assertRaises(SystemExit):
            setup.momoe_menu(a, 128)


class MacPool(unittest.TestCase):
    def test_ornith_as_before(self):
        for mem in (16, 24, 28, 64):
            self.assertEqual(setup.mac_pool_gib(mem), setup.mac_pool_gib(mem, setup.MAC_EXPERTS_GIB))

    def test_a_small_model_fits(self):
        self.assertIsNone(setup.mac_pool_gib(32, 4.4, 0.6))      # LFM2.5-8B UD-Q4_K_XL
        self.assertIsNone(setup.mac_pool_gib(64, 29.9, 6.5))     # Bolt UD-Q3_K_XL on 64 GB (measured: it ran so)

    def test_large_other_weights_leave_less(self):
        # GLM-5.3-Flash UD-IQ3_XXS (9.8 GiB of other weights) on 64 GB: under Metal's 48 GiB working set (47 GiB of
        # pool beside them: a bus error on the M1 Ultra)
        self.assertEqual(setup.mac_pool_gib(64, 102, 9.8), 35)
        self.assertEqual(setup.mac_pool_gib(32, 102, 9.8), 11)

    def test_the_kv_cache_counts(self):
        # Swift / Flash-Next IQ3_XXS on 64 GB: 40 GiB of experts + 3.95 of other weights; 1.5 GiB of KV at 64K left too
        # little (Metal out of memory, measured), so every expert fits only at 32K (0.75 GiB)
        self.assertEqual(setup.mac_pool_gib(64, 39.97, 3.95, 0, 1.5), 39)
        self.assertIsNone(setup.mac_pool_gib(64, 39.97, 3.95, 0, 0.75))

    def test_what_the_other_programs_leave(self):
        # the M1 Ultra with ~46 GB available: 46 - 9.8 - 6 = 30 GiB (35 ran it out of memory)
        self.assertEqual(setup.mac_pool_gib(64, 102, 9.8, 46.0), 30)
        self.assertEqual(setup.mac_pool_gib(64, 102, 9.8, 0.0), 35)          # unknown: the memory's rule alone
        self.assertIsNone(setup.mac_pool_gib(64, 4.4, 0.6, 46.0))             # a small model still fits whole

    def test_available_from_vm_stat(self):
        out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  1000.\nPages active: 5.\n"
               "Pages inactive:  24.\nPages speculative: 0.\nPages purgeable: 0.\n")
        with mock.patch.object(setup.subprocess, "run", return_value=mock.Mock(stdout=out)):
            self.assertAlmostEqual(setup.mac_available_gib(), 1024 * 16384 / 2**30)


class Routing(unittest.TestCase):
    def run_main(self, platform, gpus, argv):
        calls = []
        with mock.patch.object(setup.sys, "platform", platform), \
                mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "gpus", return_value=gpus), \
                mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "momoe_setup", side_effect=lambda *x: calls.append(x) or 0), \
                mock.patch.object(setup, "step", side_effect=SystemExit("Strata's path")), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--setup", "--yes", "--no-start"] + argv), \
                mock.patch.dict(os.environ, {"MOMOE_MAC_MEMORY_GIB": "64"}):
            try:
                setup.main()
            except SystemExit:
                pass
        return calls

    SPARK = [{"index": 0, "name": "NVIDIA GB10", "vram_gb": 119.0, "arch": "121", "driver": "580", "unified": True}]
    CARD = [{"index": 0, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]

    def test_a_mac_takes_any_family(self):
        calls = self.run_main("darwin", [], ["--family", "qwen"])
        self.assertEqual(len(calls), 1)

    def test_the_spark_takes_a_qwen_family(self):
        calls = self.run_main("linux", self.SPARK, ["--family", "ornith"])
        self.assertEqual(len(calls), 1)

    def test_a_card_keeps_strata_for_qwen(self):
        self.assertEqual(self.run_main("linux", self.CARD, ["--family", "ornith"]), [])

    def test_equal_cards_the_newer(self):
        two = [{"index": 0, "name": "RTX 3060", "vram_gb": 12.0, "arch": "86", "driver": "580", "unified": False},
               {"index": 1, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]
        seen = []
        with mock.patch.object(setup, "gpus", return_value=two), mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup, "ram_gb", return_value=63.0), \
                mock.patch.object(setup, "say", side_effect=lambda *x: seen.append(" ".join(map(str, x)))), \
                mock.patch.object(setup, "momoe_menu", side_effect=SystemExit("stop")):
            with self.assertRaises(SystemExit):
                setup.momoe_setup(mock.Mock(gpu=None, gpus=None, family=None), Path("."), 8080)
        self.assertTrue(any("4070 Ti" in line for line in seen), seen)

    def run_setup(self, gpus, argv_extra, family):
        """momoe_setup up to the config (no build, no download): the config it would write."""
        written = {}
        a = mock.Mock(gpu=None, gpus=None, family=family, model=None, context=None, host=None, api_key=None,
                      config_name=None, no_start=True, yes=True, draft_vocab=None, models_dir="/m", helper_tier=False,
                      layer_split=None, backend=None, vision=None, kv=None)
        for k, v in argv_extra.items():
            setattr(a, k, v)
        with mock.patch.object(setup, "gpus", return_value=gpus), mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "WIN", False), mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup, "ram_gb", return_value=63.0), mock.patch.object(setup, "say"), \
                mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), mock.patch.object(setup, "step"), \
                mock.patch.object(setup, "find_nvcc", return_value=("/nvcc", (13, 0))), \
                mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "run"), \
                mock.patch.object(setup.Path, "exists", return_value=True), mock.patch.object(setup, "done", return_value=True), \
                mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "check_shards"), \
                mock.patch.object(setup, "weights_gib", return_value=(100.0, 7.0)), \
                mock.patch.object(setup, "name_config", side_effect=lambda p, a: p), \
                mock.patch.object(setup, "write_setup_config", side_effect=lambda p, c: written.update(c)):
            setup.momoe_setup(a, Path("/d"), 8080, family)
        return written

    TWO = [{"index": 0, "name": "RTX 3060", "vram_gb": 12.0, "arch": "86", "driver": "580", "unified": False},
           {"index": 1, "name": "RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580", "unified": False}]

    def test_glm_on_two_12_gb_cards_splits(self):
        cfg = self.run_setup(self.TWO, {}, "glm53")
        self.assertEqual(cfg["gpu"], [0, 1])                 # (the server stages the faster card last under "auto")
        self.assertNotIn("--helper-tier", cfg["args"])

    def test_one_card_stays_one(self):
        self.assertEqual(self.run_setup(self.TWO, {}, "lfm25")["gpu"], 1)    # the newer of two equal cards

    def test_gpus_and_helper_tier(self):
        cfg = self.run_setup(self.TWO, {"gpus": "1,0", "helper_tier": True}, "lfm25")
        self.assertEqual(cfg["gpu"], [1, 0])
        self.assertIn("--helper-tier", cfg["args"])
        cfg = self.run_setup(self.TWO, {"gpus": "0,1", "layer_split": "18"}, "bolt")
        self.assertEqual((cfg["gpu"], cfg["layer_split"]), ([0, 1], "18"))

    def test_a_card_takes_a_ggml_family(self):
        calls = self.run_main("linux", self.CARD, ["--family", "lfm25"])
        self.assertEqual(calls[0][3], "lfm25")


if __name__ == "__main__":
    unittest.main()
