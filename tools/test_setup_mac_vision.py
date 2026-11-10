"""tools/test_setup_mac_vision.py - pictures on a Mac before macOS 14 are off, with the reason: Metal has no BF16 there,
the GPU image encoder stopped at its first picture (and with it the server), and on the CPU one picture took over 300 s
(an 8 GB M1, macOS 11.7).  No question is asked and nothing is downloaded; macOS 14 and newer ask as before."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

FAM = setup.GGML_FAMILIES["gemma4"]


def vision(version, asked):
    a = SimpleNamespace(vision=asked, yes=True, gguf_dir=None, vision_tokens=None)
    with mock.patch.object(setup.sys, "platform", "darwin"), \
            mock.patch.object(setup.platform, "mac_ver", return_value=(version, ("", "", ""), "arm64")), \
            mock.patch.object(setup, "ask", return_value="n") as ask, \
            mock.patch.object(setup, "download") as download, \
            mock.patch.object(setup, "say"), mock.patch.object(setup, "warn") as warn:
        out = setup.momoe_vision(a, "gemma4", FAM, Path("/nowhere"), Path("/nowhere/m.gguf"))
    return out, ask.called, download.called, warn.called


class MacVision(unittest.TestCase):
    def test_macos_11_to_13_no_pictures_and_no_question(self):
        for v in ("11.7.10", "12.7.6", "13.6"):
            with self.subTest(v=v):
                self.assertEqual(vision(v, None), (None, False, False, False))

    def test_asked_for_them_anyway_a_warning(self):
        self.assertEqual(vision("12.7.6", "yes"), (None, False, False, True))

    def test_macos_14_and_newer_ask_as_before(self):
        for v in ("14.0", "15.6.1", "27.0"):
            with self.subTest(v=v):
                out, asked, downloaded, _ = vision(v, None)
                self.assertTrue(asked)
                self.assertIsNone(out)                 # (the answer was no)

    def test_an_unreadable_version_is_not_taken_for_an_old_mac(self):
        self.assertTrue(vision("", None)[1])

    def test_mac_major(self):
        with mock.patch.object(setup.platform, "mac_ver", return_value=("11.7.10", ("", "", ""), "arm64")):
            self.assertEqual(setup.mac_major(), 11)
        with mock.patch.object(setup.platform, "mac_ver", return_value=("", ("", "", ""), "")):
            self.assertEqual(setup.mac_major(), 0)


if __name__ == "__main__":
    unittest.main()
