"""Setup's MTP download and the Gyro low-RAM pack (#3193 #3214 #3257): the draft layer's download is run again when it stops, every tool
setup runs ends with setup, and Gyro's low-RAM pack gets --compat-bf16 like its first pack.  Everything mocked; nothing downloaded.

    python -m unittest tools.test_setup_dl_fix
"""
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_unsloth as U  # noqa: E402


class FetchMtp(unittest.TestCase):
    def run_with(self, codes):
        calls = []

        def fake_run(cmd, cwd=None, env=None, check=True, quiet=False):
            calls.append((cmd, env, check))
            return mock.Mock(returncode=codes[min(len(calls) - 1, len(codes) - 1)])
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup.time, "sleep", lambda s: None), \
                mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "warn", lambda *a, **k: None):
            try:
                setup.fetch_mtp(pathlib.Path("mtp"), {"A": "1"})
                result = "ok"
            except SystemExit as e:
                result = e.code
        return result, calls

    def test_a_stopped_download_is_run_again_and_continues(self):
        result, calls = self.run_with([1, 1, 0])
        self.assertEqual((result, len(calls)), ("ok", 3))
        self.assertTrue(all(c[0][2] == "fetch" and c[1]["A"] == "1" and c[2] is False for c in calls))

    def test_a_download_that_never_works_ends_setup_with_a_hint(self):
        out = []
        with mock.patch.object(setup, "fail", lambda msg, hint=None: out.append((msg, hint)) or (_ for _ in ()).throw(SystemExit(1))):
            result, calls = self.run_with([1])
        self.assertEqual((result, len(calls)), (1, 3))
        self.assertIn("modelscope", out[0][1])

    def test_the_source_reaches_the_fetch(self):
        with mock.patch.object(setup, "model_source", lambda: "modelscope"):
            _, calls = self.run_with([0])
        self.assertEqual(calls[0][1]["STRATA_SOURCE"], "modelscope")


class RunEndsWithSetup(unittest.TestCase):
    def test_tools_are_told_to_end_when_setup_ends(self):
        seen = {}

        def fake_subprocess_run(cmd, **kw):
            seen.update(kw)
            return mock.Mock(returncode=0)
        with mock.patch.object(setup.subprocess, "run", fake_subprocess_run), mock.patch.object(setup, "say", lambda *a, **k: None), \
                mock.patch.object(setup, "acquisition", mock.Mock(), create=True):   # (Momoe: no Control acquisition policy)
            setup.run([sys.executable, "x.py"])
            self.assertEqual(seen["env"]["STRATA_EXIT_WITH_PARENT"], "1")
            self.assertIn("PATH", seen["env"] if os.name != "nt" else {k.upper(): v for k, v in seen["env"].items()})
            setup.run([sys.executable, "x.py"], env={"B": "2", "STRATA_EXIT_WITH_PARENT": "0"})
            self.assertEqual(seen["env"]["STRATA_EXIT_WITH_PARENT"], "0", "a value the caller set is kept")
            self.assertEqual(seen["env"]["B"], "2")


class GyroLowRam(U.Base):
    @unittest.skip("Momoe's setup has no Gyro family (the low-RAM pack's --compat-bf16 is in test_setup_qwen36)")
    def test_the_low_ram_pack_gets_compat_bf16_like_the_first_pack(self):
        code, out, _ = self.main(["--family", "gyro", "--low-ram", "on", "--context", "8192"], m="Gyro-S", family=False, ram=31.9, version="0.1.45.1")
        packs = [r for r in self.runs if any(x.endswith("iq_pack.py") for x in r)]
        self.assertTrue(packs, (code, out[-1500:]))
        self.assertTrue(all("--compat-bf16" in r for r in packs), packs)
        self.assertTrue(any("--experts-bin" in r for r in packs), "the low-RAM pack ran")


if __name__ == "__main__":
    unittest.main()
