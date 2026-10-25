"""Tests for setup.py's pictures on Momoe's ggml engine (no GPU, no network, nothing installed): which families have an
image encoder, and that images stay off unless asked for.

    python -m unittest tools.test_setup_momoe_vision
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Encoders(unittest.TestCase):
    def test_the_ggml_families_with_an_encoder_pin_it(self):
        for family in ("bolt", "glm53", "ornith122", "gemma4"):
            url, name, size, sha = setup.momoe_mmproj(family, setup.GGML_FAMILIES[family])
            self.assertTrue(url.endswith("/" + name), family)
            self.assertGreater(size, 5e8, family)
            self.assertEqual(len(sha), 64, family)

    def test_text_only_families_have_none(self):
        for family in ("flint", "lightning", "lfm25", "lfm25-8b"):
            self.assertIsNone(setup.momoe_mmproj(family, setup.GGML_FAMILIES[family]), family)

    def test_the_qwen_families_use_their_own(self):
        url, name, _, _ = setup.momoe_mmproj("qwen", setup.FAMILIES["qwen"])
        self.assertEqual(name, setup.FAMILIES["qwen"]["mmproj"])
        self.assertTrue(url.endswith(name))


class OptIn(unittest.TestCase):
    def run_vision(self, family, vision=None, yes=True, answer=None):
        a = types.SimpleNamespace(vision=vision, yes=yes)
        fam = setup.GGML_FAMILIES.get(family) or setup.FAMILIES[family]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(setup, "download", side_effect=AssertionError("no download")), \
                mock.patch.object(setup, "build_momoe_vision", side_effect=AssertionError("no build")), \
                mock.patch.object(setup, "ask", return_value=answer or "n") as asked:
            return quiet(setup.momoe_vision, a, family, fam, Path(d), Path(d) / "model.gguf"), asked

    def test_yes_alone_keeps_images_off(self):
        got, _ = self.run_vision("gemma4")
        self.assertIsNone(got)

    def test_no_turns_them_off_without_asking(self):
        got, asked = self.run_vision("gemma4", vision="no", yes=False)
        self.assertIsNone(got)
        asked.assert_not_called()

    def test_a_text_only_family_says_so_and_stays_text(self):
        got, asked = self.run_vision("flint", vision="yes")
        self.assertIsNone(got)
        asked.assert_not_called()

    def test_yes_writes_the_vision_section(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            name, size, sha = setup.GGML_FAMILIES["gemma4"]["mmproj"]
            a = types.SimpleNamespace(vision="yes", yes=True)
            with mock.patch.object(setup, "done", return_value=True), \
                    mock.patch.object(setup, "verify_sha256") as verify, \
                    mock.patch.object(setup, "build_momoe_vision", return_value=Path("/x/momoe-vision")):
                (folder / name).write_bytes(b"")
                v = quiet(setup.momoe_vision, a, "gemma4", setup.GGML_FAMILIES["gemma4"], folder, folder / "m.gguf")
            verify.assert_called_once_with(folder / name, size, sha)
        self.assertEqual(v["mmproj"], str(folder / name))
        self.assertEqual(v["model"], str(folder / "m.gguf"))
        self.assertFalse(v["gpu"])
        self.assertLessEqual(v["max_tokens"], 512)            # a Gemma 4 picture is read in one batch (--batch 512)
        self.assertNotIn("min_tokens", v)

    def test_the_qwen36_architecture_gets_1024_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            name = setup.GGML_FAMILIES["ornith122"]["mmproj"][0]
            (folder / name).write_bytes(b"")
            a = types.SimpleNamespace(vision="yes", yes=True)
            with mock.patch.object(setup, "done", return_value=True), mock.patch.object(setup, "verify_sha256"), \
                    mock.patch.object(setup, "build_momoe_vision", return_value=Path("/x/momoe-vision")):
                v = quiet(setup.momoe_vision, a, "ornith122", setup.GGML_FAMILIES["ornith122"], folder, folder / "m.gguf")
        self.assertEqual((v["min_tokens"], v["max_tokens"]), (1024, 1024))

    def vision_section(self, family, tokens):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            name = setup.GGML_FAMILIES[family]["mmproj"][0]
            (folder / name).write_bytes(b"")
            a = types.SimpleNamespace(vision="yes", yes=True, vision_tokens=tokens)
            with mock.patch.object(setup, "done", return_value=True), mock.patch.object(setup, "verify_sha256"), \
                    mock.patch.object(setup, "build_momoe_vision", return_value=Path("/x/momoe-vision")):
                return quiet(setup.momoe_vision, a, family, setup.GGML_FAMILIES[family], folder, folder / "m.gguf")

    def test_vision_tokens(self):
        self.assertEqual(self.vision_section("bolt", 600)["max_tokens"], 600)
        self.assertEqual(self.vision_section("bolt", None)["max_tokens"], 300)
        v = self.vision_section("ornith122", 2048)                     # the Qwen3.6 architecture: at least 1,024
        self.assertEqual((v["min_tokens"], v["max_tokens"]), (1024, 2048))
        v = self.vision_section("ornith122", 300)
        self.assertEqual((v["min_tokens"], v["max_tokens"]), (1024, 1024))
        self.assertEqual(self.vision_section("gemma4", 1024)["max_tokens"], 512)   # one batch of 512
        with self.assertRaises(SystemExit):
            self.vision_section("bolt", 0)

if __name__ == "__main__":
    unittest.main()
