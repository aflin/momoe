"""tools/test_setup_release.py - Momoe's Linux release (docker/): the CUDA 13 runtime its engines load is this PC's own
(the loader's cache, else a CUDA 13 toolkit), else NVIDIA's pip packages, else setup stops and says what to install;
the found folders go into the release engines' BUILD.json files and nowhere else; the release's momoe-cuda is used
only for this source and for cards it has code for."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

LDCONFIG_ALL = """3 libs found in cache `/etc/ld.so.cache'
\tlibcudart.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudart.so.13
\tlibcublas.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublas.so.13
\tlibcublasLt.so.13 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublasLt.so.13
"""
LDCONFIG_12 = """\tlibcudart.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudart.so.12
\tlibcublas.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcublas.so.12
"""


def libs(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in setup.CUDA_RUNTIME_LIBS:
        (d / n).write_bytes(b"")
    return d


class SystemRuntime(unittest.TestCase):
    def test_the_loaders_cache(self):
        with mock.patch.object(setup, "out", return_value=LDCONFIG_ALL), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, None)):
            self.assertEqual(setup.system_cuda13_dirs(), [])

    def test_cuda_12_alone_is_not_it(self):
        with mock.patch.object(setup, "out", return_value=LDCONFIG_12), \
                mock.patch.object(setup, "find_nvcc", return_value=(None, None)):
            self.assertIsNone(setup.system_cuda13_dirs())

    def test_a_toolkit_off_the_loaders_path(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "cuda-13.0"
            (root / "bin").mkdir(parents=True)
            (root / "bin" / "nvcc").write_bytes(b"")
            lib = libs(root / "targets" / "x86_64-linux" / "lib")
            with mock.patch.object(setup, "out", return_value=LDCONFIG_12), \
                    mock.patch.object(setup, "find_nvcc", return_value=(str(root / "bin" / "nvcc"), (13, 0))):
                self.assertEqual(setup.system_cuda13_dirs(), [str(lib)])

    def test_a_cuda_12_toolkit_is_not_it(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "cuda-12.9"
            (root / "bin").mkdir(parents=True)
            (root / "lib64").mkdir()
            (root / "lib64" / "libcudart.so.12").write_bytes(b"")
            with mock.patch.object(setup, "out", return_value=""), \
                    mock.patch.object(setup, "find_nvcc", return_value=(str(root / "bin" / "nvcc"), (12, 9))):
                self.assertIsNone(setup.system_cuda13_dirs())


class ReleaseRuntime(unittest.TestCase):
    """release_cuda_dirs in a release tree (a temporary ROOT with the three stamps)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stamps = [self.root / "engine" / "BUILD.json", self.root / "engine-dsv41" / "BUILD.json",
                       self.root / "build-cuda" / "BUILD.json"]
        for s in self.stamps:
            s.parent.mkdir()
            s.write_text(json.dumps({"source": "local", "release": True, "cuda_dirs": []}))
        self.patches = [mock.patch.object(setup, "ROOT", self.root), mock.patch.object(setup, "_RELEASE_DIRS", None),
                        mock.patch.object(setup, "engine_dir", lambda tk=13: self.root / "engine"),
                        mock.patch.object(setup, "gpus", return_value=[{"index": 0, "driver": "580.95.05"}])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def dirs_in_stamps(self):
        return [json.loads(s.read_text())["cuda_dirs"] for s in self.stamps]

    def test_the_pcs_own_runtime_first(self):
        with mock.patch.object(setup, "system_cuda13_dirs", return_value=[]), \
                mock.patch.object(setup, "pip_install") as pip:
            self.assertEqual(setup.release_cuda_dirs(), [])
        pip.assert_not_called()

    def test_pip_when_the_pc_has_none(self):
        wheels = libs(self.root / "venv" / "nvidia" / "cu13" / "lib")
        with mock.patch.object(setup, "system_cuda13_dirs", return_value=None), \
                mock.patch.object(setup, "pip_install") as pip, \
                mock.patch.object(setup, "cuda_lib_dirs", return_value=[str(wheels)]):
            self.assertEqual(setup.release_cuda_dirs(), [str(wheels)])
        pip.assert_called_once()
        self.assertEqual(pip.call_args[0][0], setup.CUDA_WHEELS)
        self.assertEqual(self.dirs_in_stamps(), [[str(wheels)]] * 3)

    def test_instructions_when_pip_fails(self):
        with mock.patch.object(setup, "system_cuda13_dirs", return_value=None), \
                mock.patch.object(setup, "pip_install", side_effect=SystemExit(1)), \
                mock.patch.object(setup, "cuda_lib_dirs", return_value=[]), \
                mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as fail:
            with self.assertRaises(SystemExit):
                setup.release_cuda_dirs()
        what, hint = fail.call_args[0]
        self.assertIn("libcudart.so.13", what)
        self.assertIn("cuda-libraries-13-0", hint)
        self.assertIn("pip install " + " ".join(setup.CUDA_WHEELS), hint)

    def test_only_the_release_stamps_change(self):
        own = self.stamps[1]                       # DeepSeek's engine compiled on this PC: its toolkit stays
        own.write_text(json.dumps({"source": "local", "cuda_dirs": ["/usr/local/cuda-13.0/lib64"]}))
        with mock.patch.object(setup, "system_cuda13_dirs", return_value=["/opt/cuda/lib64"]):
            setup.release_cuda_dirs()
        self.assertEqual(self.dirs_in_stamps(), [["/opt/cuda/lib64"], ["/usr/local/cuda-13.0/lib64"], ["/opt/cuda/lib64"]])

    def test_a_driver_too_old_for_cuda_13(self):
        with mock.patch.object(setup, "gpus", return_value=[{"index": 0, "driver": "570.86.10"}]), \
                mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as fail:
            with self.assertRaises(SystemExit):
                setup.release_cuda_dirs()
        self.assertIn("580", fail.call_args[0][0])


class ReleaseGgml(unittest.TestCase):
    def ok(self, meta, archs, exe=True):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t) / "build-cuda"
            b.mkdir()
            (b / "BUILD.json").write_text(json.dumps(meta))
            if exe:
                (b / "momoe-cuda").write_bytes(b"")
            with mock.patch.object(setup, "ROOT", Path(t)), mock.patch.object(setup, "source_hash", return_value="abc"):
                return setup.release_ggml_ok(archs)

    def test_the_release_for_its_cards(self):
        meta = {"src": "abc", "archs": "75;80;86;89;90;100;120", "release": True}
        self.assertTrue(self.ok(meta, {"89"}))
        self.assertTrue(self.ok(meta, {"86", "120"}))

    def test_a_card_it_has_no_code_for(self):
        self.assertFalse(self.ok({"src": "abc", "archs": "75;80;86;89", "release": True}, {"121"}))

    def test_another_source(self):                 # a git pull since: compiled here
        self.assertFalse(self.ok({"src": "old", "archs": "89", "release": True}, {"89"}))

    def test_one_compiled_here_is_not_the_release(self):
        self.assertFalse(self.ok({"src": "abc", "archs": "89"}, {"89"}))

    def test_the_macs_momoe_metal(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t) / "build-metal"
            b.mkdir()
            (b / "BUILD.json").write_text(json.dumps({"src": "abc", "release": True}))
            (b / "momoe-metal").write_bytes(b"")
            with mock.patch.object(setup, "ROOT", Path(t)), mock.patch.object(setup, "source_hash", return_value="abc"), \
                    mock.patch.object(setup.sys, "platform", "darwin"):
                self.assertTrue(setup.release_ggml_ok())
                (b / "BUILD.json").write_text(json.dumps({"src": "abc"}))      # compiled on this Mac: not the release's
                self.assertFalse(setup.release_ggml_ok())

    def test_no_engine(self):
        self.assertFalse(self.ok({"src": "abc", "archs": "89", "release": True}, {"89"}, exe=False))


class ReleaseEngine(unittest.TestCase):
    def test_the_engines_runtime_folders_for_its_own_programs(self):
        # (`momoe --families` runs before the model starts: with only the pip packages it must find them)
        with tempfile.TemporaryDirectory() as t:
            eng, wheels = Path(t) / "engine", libs(Path(t) / "nvidia" / "cu13" / "lib")
            eng.mkdir()
            (eng / "BUILD.json").write_text(json.dumps({"release": True, "cuda_dirs": [str(wheels), "/gone"]}))
            with mock.patch.dict(setup.os.environ, {"LD_LIBRARY_PATH": "/x"}):
                self.assertEqual(setup.engine_env(eng)["LD_LIBRARY_PATH"], str(wheels) + setup.os.pathsep + "/x")

    def test_build_engine_finds_the_runtime_for_the_release_engine(self):
        with tempfile.TemporaryDirectory() as t:
            eng = Path(t) / "engine"
            eng.mkdir()
            (eng / setup.EXE).write_bytes(b"")
            meta = {"source": "local", "release": True, "archs": [89], "src": "s", "version": "1", "cuda_dirs": []}
            (eng / "BUILD.json").write_text(json.dumps(meta))
            with mock.patch.object(setup, "engine_dir", return_value=eng), \
                    mock.patch.object(setup, "source_hash", return_value="s"), \
                    mock.patch.object(setup, "stale_binary", return_value=False), \
                    mock.patch.object(setup, "cpu_floor", return_value=""), \
                    mock.patch.object(setup, "cpu_info", return_value=("cpu", True)), \
                    mock.patch.object(setup, "release_cuda_dirs") as rt, \
                    mock.patch.object(setup, "install_build_tools") as tools:
                self.assertEqual(setup.build_engine({"arch": "89"}, "none", True, None, toolkit=13), eng)
        rt.assert_called_once_with(True)
        tools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
