"""Momoe on a Mac: the expert pool setup gives momoe-metal for the Mac's memory (Ornith-1.5 IQ4_XS).

    python -m unittest tools.test_setup_mac
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


class Pool(unittest.TestCase):
    def test_pool_by_memory(self):
        # 16 GB: 5 GiB (measured: an 8.6 GB server); every expert (15.9 GiB; the MTP layer's with the other weights)
        # in memory from ~28 GB (Metal's working set, 75%, less the other weights and 3 GiB for the compute buffers)
        for mem, pool in ((16, 5), (18, 7), (24, 13), (26, 14), (27, 15), (28, None), (32, None), (64, None)):
            self.assertEqual(setup.mac_pool_gib(mem), pool, mem)

    def test_never_below_3(self):
        self.assertEqual(setup.mac_pool_gib(13), 3)

    def test_memory_override(self):
        with mock.patch.dict(os.environ, {"MOMOE_MAC_MEMORY_GIB": "16"}):
            self.assertEqual(setup.mac_memory_gib(), 16.0)


if __name__ == "__main__":
    unittest.main()
