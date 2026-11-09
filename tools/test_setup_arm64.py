"""tools/test_setup_arm64.py - setup.py's CPU probe on an ARM64 machine (DGX Spark / GB10, Grace; upstream #3622, whose
setup.py part Momoe had not taken): /proc/cpuinfo there has no x86 "flags" line, which stopped cpu_info() with an
AttributeError; lscpu names the cores; there is no older-CPU floor (it read "unsupported").  x86 keeps its old answers."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

ARM_CPUINFO = ("processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics\n"
               "CPU implementer\t: 0x41\n")
SPARK_LSCPU = ("Architecture:             aarch64\n  Model name:             Cortex-X925\n    Thread(s) per core:   1\n"
               "  Model name:             Cortex-A725\n    Thread(s) per core:   1\n")


class ArmCpu(unittest.TestCase):
    def test_an_arm_cpuinfo_without_the_x86_flags_line_does_not_crash(self):
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=ARM_CPUINFO)), \
                mock.patch.object(setup, "out", return_value=SPARK_LSCPU):
            self.assertEqual(setup.cpu_info(), ("Cortex-X925 + Cortex-A725", False, False))

    def test_a_name_without_lscpu(self):
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=ARM_CPUINFO)), \
                mock.patch.object(setup, "out", return_value=""), mock.patch.object(setup.platform, "processor", return_value=""):
            self.assertEqual(setup.cpu_info(), ("arm64 CPU", False, False))

    def test_an_x86_cpuinfo_is_read_as_before(self):
        text = "flags\t\t: fpu avx avx2 avx512f avx512bw avx512vl avx512_vnni avx512vbmi\nmodel name\t: Test CPU\n"
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=text)):
            self.assertEqual(setup.cpu_info(), ("Test CPU", True, True))

    def test_no_older_cpu_floor_on_arm(self):
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}), mock.patch.object(setup, "is_arm64", return_value=True):
            self.assertEqual(setup.cpu_floor(False), "")

    def test_the_floor_on_x86_without_avx2_is_still_asked_for(self):
        with mock.patch.dict(os.environ, {"STRATA_ISA_FLOOR": ""}), mock.patch.object(setup, "is_arm64", return_value=False), \
                mock.patch.object(setup, "WIN", False), \
                mock.patch("builtins.open", mock.mock_open(read_data="flags\t: fpu sse4_2 popcnt avx\n")):
            self.assertEqual(setup.cpu_floor(False), "avx")


if __name__ == "__main__":
    unittest.main()
