"""tools/test_setup_dsv41.py - Momoe: DeepSeek-V4.1-Flash (MULTI_FILE_FAMILIES["deepseek41"], install_dsv41, from upstream
Strata 0.2.6): each set's shards pinned with their sizes and SHA-256 at the pinned revisions, the shared files once, the
family in --family and --model, the plan's line for a 12 GB card, the pack recipe, NVIDIA only, and the config it writes
(its own engine, the pack's tokenizer, arch deepseek41, the long load time).

    python tools/test_setup_dsv41.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

FAM = setup.MULTI_FILE_FAMILIES["deepseek41"]
CARD = {"index": 1, "name": "NVIDIA GeForce RTX 4070 Ti", "vram_gb": 12.0, "arch": "89", "driver": "580"}


class Files(unittest.TestCase):
    def test_sets_pinned(self):
        sizes = {"IST-DAS-2.00bpw": 138.2, "IST-DAS-2.25bpw": 155.2, "IST-DAS-1.75bpw": 120.7}
        for m, gb in sizes.items():
            md = FAM["models"][m]
            self.assertEqual(len(md["shards"]), 4)
            self.assertAlmostEqual(sum(FAM["sha256"][s][0] for s in md["shards"]) / 1e9, gb, places=1)
            self.assertEqual(md["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK])
            for f in md["shards"] + md["shared"]:
                self.assertEqual(len(FAM["sha256"][f][1]), 64, f)
        self.assertEqual(len(FAM["models"]["UD-Q8_K_XL"]["shards"]), 9)

    def test_where_each_file_comes_from(self):
        for repo in (setup.DSV41_REPO, setup.DSV41_ISTA_REPO):
            self.assertRegex(setup.HF_REVISIONS[repo], "^[0-9a-f]{40}$")
        self.assertIn(setup.DSV41_ISTA_REPO, setup.dsv41_hf(setup.DSV41_ISTA_ENGRAM))
        self.assertTrue(setup.dsv41_hf(setup.DSV41_DSPARK).endswith(setup.HF_REVISIONS[setup.DSV41_REPO] + "/"))
        self.assertTrue(setup.dsv41_hf(setup.DSV41_SHARD.format(i=1)).endswith("/UD-Q8_K_XL/"))

    def test_choices(self):
        for m in FAM["models"]:
            self.assertIn(m, setup.model_choices())

    def test_the_shared_folder(self):
        a = mock.Mock(models_dir="/m", gguf_dir=None)
        self.assertEqual(setup.dsv41_shared_dir(a, FAM), Path("/m/deepseek-v4.1-flash-shared"))
        self.assertEqual(setup.dsv41_folder(a, FAM, "IST-DAS-2.00bpw"), Path("/m/deepseek-v4.1-flash-IST-DAS-2.00bpw"))

    def test_the_pack_recipe(self):
        cmd = setup.dsv41_pack_cmd(Path("/f"), ["/f/a.gguf", "/f/b.gguf"], Path("/s/e"), Path("/s/m"), Path("/s/d"), Path("/p"))
        self.assertEqual(cmd[cmd.index("--shards") + 1:cmd.index("--shards") + 3], ["a.gguf", "b.gguf"])
        self.assertEqual(cmd[cmd.index("--dspark-noise-token") + 1], "128799")
        self.assertNotIn("--skip-tensors", cmd)                 # (packs without an error and gives garbage text)
        self.assertNotIn("--dense-override", cmd)


class Smoke(unittest.TestCase):
    def test_the_cards_as_nvidia_smi_numbers_them(self):
        """GPU 1 of nvidia-smi (an RTX 4070 Ti beside a 3060) is CUDA's 0 in its own order: the test run sees only that card."""
        seen = {}
        def run(cmd, **kw):
            seen.update(cmd=cmd, env=kw["env"])
            return mock.Mock(returncode=0, stdout="Paris", stderr="")
        with mock.patch.object(setup.subprocess, "run", side_effect=run):
            setup.dsv41_smoke(Path("/e/x"), Path("/p"), [], [1])
        self.assertEqual((seen["env"]["CUDA_DEVICE_ORDER"], seen["env"]["CUDA_VISIBLE_DEVICES"]), ("PCI_BUS_ID", "1"))
        self.assertNotIn("--gpus", seen["cmd"])


class Plan(unittest.TestCase):
    def test_a_12_gb_card_and_63_gb(self):
        from strata_control import dsv41_fit
        self.assertEqual(dsv41_fit.recommend(63.0, 326.0), "IST-DAS-2.00bpw")
        p = dsv41_fit.plan([CARD], 63.0, 326.0, 8192, model="IST-DAS-2.00bpw")
        self.assertEqual(p["args"], [])
        self.assertTrue(any("Momoe keeps the experts" in line for line in p["lines"]))


class Install(unittest.TestCase):
    def run_setup(self, hip=False, **extra):
        """install_dsv41 to its config (no build, no download, no test run): the config it would write."""
        written = {}
        a = mock.Mock(model="IST-DAS-2.00bpw", context=None, gguf_dir=None, models_dir="/d/models", vision=None,
                      check=False, yes=True, config_name=None, host=None, api_key=None, no_start=True)
        for k, v in extra.items():
            setattr(a, k, v)
        with mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), mock.patch.object(setup, "warn"), \
                mock.patch.object(setup, "step"), mock.patch.object(setup, "pip_install"), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/llama")), \
                mock.patch.object(setup, "build_dsv41_engine", return_value=(Path("/e/strata-dsv41"), ["/cuda/bin"])), \
                mock.patch.object(setup, "free_gb", return_value=1000.0), mock.patch.object(setup, "done", return_value=True), \
                mock.patch.object(setup.Path, "exists", return_value=True), \
                mock.patch.object(setup.Path, "is_file", return_value=False), \
                mock.patch.object(setup, "download", side_effect=AssertionError("downloaded")), \
                mock.patch.object(setup, "verify_sha256"), mock.patch.object(setup, "run"), \
                mock.patch.object(setup, "dsv41_pack_info", return_value=(True, "all present")), \
                mock.patch.object(setup, "dsv41_smoke", return_value=(True, "Paris")), \
                mock.patch.object(setup, "name_config", side_effect=lambda p, a: p), \
                mock.patch.object(setup, "write_setup_config", side_effect=lambda p, c: written.update(c)):
            setup.install_dsv41(a, Path("/d"), "deepseek41", {**CARD, "archs": ["89"]}, hip, [CARD], 8080, 63.0)
        return written

    def test_the_config(self):
        cfg = self.run_setup()
        a = cfg["args"]
        self.assertEqual(cfg["exe"], "/e/strata-dsv41")
        self.assertEqual(a[a.index("--pack") + 1], "/d/packs/deepseek-v4.1-flash-ist-das-2.00bpw")
        self.assertEqual(a[a.index("--max-context") + 1], "8192")
        self.assertEqual(a[a.index("--dspark") + 1], "auto")        # never off: off still loads the drafter's 7 GB
        self.assertEqual(cfg["tokenizer"], "/d/packs/deepseek-v4.1-flash-ist-das-2.00bpw/tokenizer")
        self.assertEqual(cfg["arch"], "deepseek41")
        self.assertEqual(cfg["engine_ready_s"], 3600)
        self.assertEqual(cfg["gpu"], 1)
        self.assertEqual(cfg["lib_dirs"], ["/cuda/bin"])
        self.assertEqual(cfg["model_name"], "deepseek-v4.1-flash-ist-das-2.00bpw")

    def test_amd_is_said(self):
        with mock.patch.object(setup, "say"), self.assertRaises(SystemExit):
            self.run_setup(hip=True)


class MacFamily(unittest.TestCase):
    """On a Mac (and a GPU that shares the system's memory) Momoe's ggml engine runs it: momoe_setup's family."""

    G = setup.GGML_FAMILIES["deepseek41"]

    def test_the_same_pinned_shards(self):
        for m, parts in self.G["parts"].items():
            self.assertEqual([p[0] for p in parts], FAM["models"][m]["shards"])
            self.assertEqual([(p[1], p[2]) for p in parts], [FAM["sha256"][n] for n in FAM["models"][m]["shards"]])
        self.assertEqual(self.G["extra"], [(setup.DSV41_ISTA_ENGRAM, "deepseek-v4.1-flash-shared")])
        self.assertEqual(self.G["arch"], "deepseek41")
        self.assertIn("deepseek41", setup.MOMOE_MENU)

    def run_main(self, platform, gpus):
        calls = []
        with mock.patch.object(setup.sys, "platform", platform), mock.patch.object(setup, "WIN", False), \
                mock.patch.object(setup, "gpus", return_value=gpus), mock.patch.object(setup, "gpu_problem", return_value=None), \
                mock.patch.object(setup, "momoe_setup", side_effect=lambda *x: calls.append("momoe") or 0), \
                mock.patch.object(setup, "install_dsv41", side_effect=lambda *x: calls.append("dsv41") or 0), \
                mock.patch.object(setup, "step", side_effect=SystemExit("Strata's path")), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--setup", "--yes", "--no-start", "--family", "deepseek41"]), \
                mock.patch.dict(setup.os.environ, {"MOMOE_MAC_MEMORY_GIB": "64"}):
            try:
                setup.main()
            except SystemExit:
                pass
        return calls

    def test_routing(self):
        self.assertEqual(self.run_main("darwin", []), ["momoe"])
        spark = [{"index": 0, "name": "NVIDIA GB10", "vram_gb": 119.0, "arch": "121", "driver": "580", "unified": True}]
        self.assertEqual(self.run_main("linux", spark), ["momoe"])
        card = [{**CARD, "index": 0, "unified": False}]
        self.assertEqual(self.run_main("linux", card), [])      # (the PC's own path: install_dsv41 after step 1)

    def test_the_mac_pool_leaves_room(self):
        # 123 GiB of experts, 5.74 GiB of other weights and the 3.66 GiB staging copy, 32K context: 28 GiB on 64 GB
        pool = setup.mac_pool_gib(64, 123.0, 5.74 + 3.66, 0, 3200 * 32768 / 2**30) - self.G["mac_margin_gib"]
        self.assertEqual(pool, 28)


if __name__ == "__main__":
    unittest.main()
