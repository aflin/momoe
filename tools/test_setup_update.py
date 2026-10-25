"""Tests for `setup.py --update` (#475, UPDATE.bat / update.sh): it refreshes what a start would - the Python packages,
the engine, each model's config and draft subset - and never starts the model.  Every outside effect is mocked: no
GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_update
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Update(unittest.TestCase):
    def config(self, d: Path, **extra) -> Path:
        rt = d / "mtp" / "rt"
        rt.mkdir(parents=True)
        p = d / "strata-iq3_s.json"
        p.write_text(json.dumps({"model_name": "Qwen IQ3_S", "exe": str(d / "engine" / setup.EXE),
                                 "args": ["--native", "x", "--mtp", str(rt)], **extra}), encoding="utf-8")
        return p

    def run_update(self, have, argv=()):
        a = mock.Mock(**{"build": False, **dict(argv)})
        out = io.StringIO()
        with mock.patch.object(setup, "pip_install") as pip, \
                mock.patch.object(setup, "update_installed_engine") as eng, \
                mock.patch.object(setup, "update_ggml_engines") as self.ggml, \
                mock.patch.object(setup, "refresh_draft_vocab") as dv, \
                mock.patch.object(setup, "start") as start, \
                mock.patch("subprocess.call") as call, \
                contextlib.redirect_stdout(out):
            rc = setup.update_install(have, a)
        return rc, pip, eng, dv, start, call, out.getvalue()

    def test_refreshes_without_starting(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d), draft_vocab="en")
            rc, pip, eng, dv, start, call, out = self.run_update([p])
        self.assertEqual(rc, 0)
        pip.assert_called_once()
        eng.assert_called_once_with()
        self.ggml.assert_called_once_with()                       # momoe-metal / momoe-cuda too
        dv.assert_called_once()
        self.assertEqual(dv.call_args[0][1], "en")                 # the model's own subset is kept
        start.assert_not_called()
        call.assert_not_called()
        self.assertIn("Momoe is updated", out)

    def test_build_changes_nothing(self):
        # Momoe always compiles its engine: --build (kept for old scripts) still updates it
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            rc, pip, eng, dv, *_ = self.run_update([p], {"build": True})
        self.assertEqual(rc, 0)
        eng.assert_called_once_with()
        self.assertEqual(dv.call_args[0][1], "cjk")

    def test_a_ggml_engine_names_no_strata_version(self):
        # momoe-metal / momoe-cuda carry no engine version: the closing line says none (it said 0.1.12)
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d), exe=str(Path(d) / "build-metal" / "momoe-metal"))
            rc, pip, eng, dv, start, call, out = self.run_update([p])
        self.assertEqual(rc, 0)
        self.assertIn("Momoe is updated. Start a model with", out)
        self.assertNotIn("(engine", out)

    def test_nothing_installed(self):
        rc, pip, eng, dv, start, call, out = self.run_update([])
        self.assertEqual(rc, 0)
        for m in (pip, eng, dv, start, call):
            m.assert_not_called()
        self.assertIn("momoe setup", out)                         # momoe.bat setup / ./momoe setup

    def test_a_json_that_is_no_model_config_is_skipped(self):
        """#549: a strata-*.json without "args" (not written by setup) stopped update.sh with KeyError: 'args'."""
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            other = Path(d) / "strata-notes.json"
            other.write_text(json.dumps({"note": "mine"}), encoding="utf-8")
            broken = Path(d) / "strata-cut.json"
            broken.write_text("{\"args\": [", encoding="utf-8")
            rc, pip, eng, dv, start, call, out = self.run_update([other, broken, p])
        self.assertEqual(rc, 0, out)
        self.assertIn('skipped strata-notes.json (no "exe" or "args"): it is not a Momoe model config', out)
        self.assertIn("skipped strata-cut.json (not valid JSON)", out)
        dv.assert_called_once()                                     # the real model is still refreshed
        self.assertIn("Qwen IQ3_S: up to date", out)
        self.assertIn("Momoe is updated", out)

    def test_installed_configs_lists_only_model_configs(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            (Path(d) / "strata-notes.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(setup, "ROOT", Path(d)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.installed_configs(), [p])

    def test_main_update_never_starts(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "move_configs"), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_install", return_value=0) as up, \
                    mock.patch.object(setup, "start") as start, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
        up.assert_called_once()
        start.assert_not_called()


class SettingsLine(unittest.TestCase):
    """#564: a start prints the settings it uses (the engine options without the model's paths, and the server's
    fields), so a change made by hand to strata-<model>.json shows without reading the log."""

    CFG = {"exe": "x", "host": "0.0.0.0", "port": 8081, "api_key": "secret", "gpu": [0, 1], "fit_max_tokens": True,
           "args": ["--native", "E:\\Momoe\\packs\\iq3_s", "--mtp", "/s/mtp/rt", "m.gguf", "--kv", "int8",
                    "--kv-resident", "32768", "--spec-min-p", "0.5", "--vram-reserve-mib", "2048", "--mmap-experts",
                    "--prefill", "auto"]}

    def test_summary(self):
        s = setup.settings_summary(self.CFG)
        self.assertEqual(s, "--kv int8 --kv-resident 32768 --spec-min-p 0.5 --vram-reserve-mib 2048 --mmap-experts "
                            "--prefill auto; server 0.0.0.0:8081, api key set, gpu 0,1, fit_max_tokens true")
        self.assertNotIn("secret", s)
        self.assertIn("127.0.0.1:9000", setup.settings_summary({"args": []}, 9000))

    def test_a_start_prints_it(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"")
            p = Path(d) / "strata-iq3_s.json"
            args = [x for x in self.CFG["args"] if x != "m.gguf"]           # no model file here
            p.write_text(json.dumps({**self.CFG, "exe": str(exe), "gpu": 0, "args": args}), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(setup, "gpus", lambda: []), \
                    mock.patch.object(setup, "refresh_draft_vocab"), \
                    mock.patch.object(setup, "is_wsl", lambda: False), \
                    mock.patch.object(setup.subprocess, "call", return_value=0), \
                    contextlib.redirect_stdout(out):
                setup.start(p, None, yes=True)
        text = " ".join(out.getvalue().split())
        self.assertIn("Settings (strata-iq3_s.json): --kv int8 --kv-resident 32768", text)
        self.assertIn("--vram-reserve-mib 2048", text)


class GgmlEngines(unittest.TestCase):
    """Momoe's ggml engines (momoe-metal, momoe-cuda) are compiled again by --update when src/metal changed: the source
    hash in <build>/BUILD.json.  Only a build this folder has; the build script is mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "src" / "metal").mkdir(parents=True)
        (self.root / "src" / "metal" / "momoe_metal.cpp").write_text("int main() {}\n")
        self.runs = []
        self.fail_build = False

    def cuda_build(self, archs="86;89"):
        b = self.root / "build-cuda"
        (b / "cmake").mkdir(parents=True)
        (b / "momoe-cuda").write_bytes(b"x")
        (b / "cmake" / "CMakeCache.txt").write_text(f"CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CUDA_ARCHITECTURES:STRING={archs}\n")
        return b

    def update(self, platform="linux", nvcc="/usr/local/cuda/bin/nvcc"):
        def fake_run(cmd, env=None, **k):
            self.runs.append((cmd, env))
            if self.fail_build:
                raise SystemExit(1)
        out = io.StringIO()
        with mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "run", fake_run), \
                mock.patch.object(setup, "find_nvcc", return_value=(nvcc, (13, 0))), \
                mock.patch.object(setup.sys, "platform", platform), contextlib.redirect_stdout(out):
            setup.update_ggml_engines()
        return out.getvalue()

    def test_nothing_built_nothing_compiled(self):
        self.update()
        self.assertEqual(self.runs, [])

    def test_an_unstamped_build_is_compiled_once_for_its_own_architectures(self):
        b = self.cuda_build()
        out = self.update()
        self.assertEqual(len(self.runs), 1)
        cmd, env = self.runs[0]
        self.assertTrue(cmd[-1].endswith("build_cuda.sh"), cmd)
        self.assertEqual(env["CUDA_ARCHS"], "86;89")              # not "native": the cards it was built for
        self.assertEqual(env["NVCC"], "/usr/local/cuda/bin/nvcc")
        with mock.patch.object(setup, "ROOT", self.root):
            self.assertEqual(json.loads((b / "BUILD.json").read_text())["src"], setup.source_hash(setup.GGML_ENGINE_SOURCES))
        self.assertIn("momoe-cuda compiled again", out)
        self.update()                                             # stamped: the same source is not compiled again
        self.assertEqual(len(self.runs), 1)

    def test_the_stamp_keeps_the_architectures_setup_chose(self):
        b = self.cuda_build(archs="native")                       # the CMake cache does not say (it is not kept there)
        with mock.patch.object(setup, "ROOT", self.root):
            setup.stamp_ggml_build(b, "89")                         # setup built for one card (--gpu)
        (self.root / "src" / "metal" / "momoe_metal.cpp").write_text("int main() { return 1; }\n")
        self.update()
        self.assertEqual(self.runs[0][1]["CUDA_ARCHS"], "89")
        self.assertEqual(json.loads((b / "BUILD.json").read_text())["archs"], "89")   # kept for the next update

    def test_a_changed_source_is_compiled_again(self):
        self.cuda_build()
        self.update()
        (self.root / "src" / "metal" / "momoe_metal.cpp").write_text("int main() { return 0; }\n")   # a git pull
        self.update()
        self.assertEqual(len(self.runs), 2)

    def test_a_failed_build_keeps_the_engine_and_stamps_nothing(self):
        b = self.cuda_build()
        self.fail_build = True
        out = self.update()
        self.assertIn("the installed one is kept", out)
        self.assertTrue((b / "momoe-cuda").exists())
        self.assertFalse((b / "BUILD.json").exists())

    def test_no_nvcc_keeps_the_engine(self):
        self.cuda_build()
        out = self.update(nvcc=None)
        self.assertEqual(self.runs, [])
        self.assertIn("no CUDA toolkit", out)

    def test_metal_only_on_a_mac(self):
        b = self.root / "build-metal"
        b.mkdir()
        (b / "momoe-metal").write_bytes(b"x")
        self.update(platform="linux")
        self.assertEqual(self.runs, [])
        self.update(platform="darwin")
        self.assertEqual(len(self.runs), 1)
        self.assertTrue(self.runs[0][0][-1].endswith("build.sh"))


if __name__ == "__main__":
    unittest.main()
