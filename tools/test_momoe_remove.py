"""tools/test_momoe_remove.py - `momoe remove CONFIG` (and `momoe remove FOLDER`, a download no config uses): the model's folder in models/, its pack in packs/, the config and
its logs deleted after asking; what another config uses kept unless the second question or --wipe takes those configs
and their files too (as far as the sharing goes); nothing outside models/ and packs/ touched; refused while a model it
would delete runs, and without a terminal unless -y or --wipe."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import momoe  # noqa: E402


class Remove(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.t = Path(self.d.name).resolve()
        self.ps = [mock.patch.object(momoe, "ROOT", self.t), mock.patch.object(momoe, "STATE_DIR", self.t / ".momoe"),
                   mock.patch.object(momoe, "STATE", self.t / ".momoe" / "server.json"),
                   mock.patch.object(momoe, "say", lambda *a, **k: None)]
        for p in self.ps:
            p.start()
        (self.t / "config").mkdir()
        (self.t / "logs").mkdir()
        self.outside = self.t / "data" / "expert-profile.bin"     # shared data: never deleted
        self.outside.parent.mkdir()
        self.outside.write_bytes(b"x")
        self.cfg_a = self.model("a", "gemma4-UD-IQ3_XXS", "gemma4-ud-iq3_xxs")
        self.cfg_b = self.model("b", "shared-Q4", "shared-q4")
        self.cfg_c = self.model("c", "shared-Q4", "shared-q4")     # c uses b's model and pack too

    def tearDown(self):
        for p in self.ps:
            p.stop()
        self.d.cleanup()

    def model(self, name, mdir, pack):
        m = self.t / "models" / mdir
        m.mkdir(parents=True, exist_ok=True)
        (m / "model.gguf").write_bytes(b"\0" * 1000)
        (m / "model.gguf.done").write_text("")
        k = self.t / "packs" / pack / "tokenizer"
        k.mkdir(parents=True, exist_ok=True)
        (k / "vocab.json").write_text("{}")
        log = self.t / "logs" / f"{name}.log"
        log.write_text("log")
        cfg = self.t / "config" / f"{name}.json"
        cfg.write_text(json.dumps({"exe": "/x/momoe-cuda", "setup_tag": mdir, "cwd": str(self.t),
                                   "args": ["-m", str(m / "model.gguf"), "--profile", str(self.outside)],
                                   "tokenizer": str(k), "log": str(log)}))
        return cfg

    def test_deletes_model_pack_config_logs(self):
        self.assertEqual(momoe.cmd_remove(["a", "--yes"]), 0)
        self.assertFalse((self.t / "models" / "gemma4-UD-IQ3_XXS").exists())
        self.assertFalse((self.t / "packs" / "gemma4-ud-iq3_xxs").exists())
        self.assertFalse(self.cfg_a.exists())
        self.assertFalse((self.t / "logs" / "a.log").exists())
        self.assertTrue(self.outside.exists())
        self.assertTrue(self.cfg_b.exists() and (self.t / "models" / "shared-Q4" / "model.gguf").exists())

    def test_keeps_what_another_config_uses(self):
        self.assertEqual(momoe.cmd_remove(["b", "--yes"]), 0)
        self.assertFalse(self.cfg_b.exists())
        self.assertTrue((self.t / "models" / "shared-Q4" / "model.gguf").exists())
        self.assertTrue((self.t / "packs" / "shared-q4").exists())
        self.assertEqual(momoe.cmd_remove(["c", "--yes"]), 0)        # the last one using it: now it goes
        self.assertFalse((self.t / "models" / "shared-Q4").exists())

    def test_dry_run_and_no(self):
        self.assertEqual(momoe.cmd_remove(["a", "--dry-run"]), 0)
        self.assertTrue(self.cfg_a.exists())
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: True), mock.patch("builtins.input", lambda _: ""):
            self.assertEqual(momoe.cmd_remove(["a"]), 1)
        self.assertTrue((self.t / "models" / "gemma4-UD-IQ3_XXS" / "model.gguf").exists())
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: True), mock.patch("builtins.input", lambda _: "y"):
            self.assertEqual(momoe.cmd_remove(["a"]), 0)
        self.assertFalse(self.cfg_a.exists())

    def test_no_terminal_needs_yes(self):
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: False), self.assertRaises(SystemExit):
            momoe.cmd_remove(["a"])
        self.assertTrue(self.cfg_a.exists())

    def test_refused_while_running(self):
        with mock.patch.object(momoe, "running", lambda: {"pid": 1, "config": str(self.cfg_a), "model": "a"}), \
                self.assertRaises(SystemExit):
            momoe.cmd_remove(["a", "--yes"])
        self.assertTrue(self.cfg_a.exists())

    def test_delete_is_the_same_command(self):
        self.assertIs(momoe.main.__globals__["cmd_remove"], momoe.cmd_remove)
        self.assertEqual(momoe.main(["delete", "a", "--yes"]), 0)
        self.assertFalse(self.cfg_a.exists())

    def test_folder_without_config(self):
        lone = self.t / "models" / "gemma4-UD-Q4_K_XL"
        lone.mkdir()
        (lone / "model.gguf").write_bytes(b"\0" * 100)
        pack = self.t / "packs" / "gemma4-ud-q4_k_xl" / "tokenizer"
        pack.mkdir(parents=True)
        self.assertEqual(momoe.cmd_remove(["gemma4-UD-Q4_K_XL", "--dry-run"]), 0)
        self.assertTrue(lone.exists())
        self.assertEqual(momoe.main(["delete", "gemma4-ud-q4_k_xl", "--yes"]), 0)   # (any case)
        self.assertFalse(lone.exists())
        self.assertFalse(pack.parent.exists())
        self.assertTrue((self.t / "models" / "gemma4-UD-IQ3_XXS" / "model.gguf").exists())

    def test_folder_a_config_uses_is_refused(self):
        with self.assertRaises(SystemExit):
            momoe.cmd_remove(["shared-Q4", "--yes"])
        self.assertTrue((self.t / "models" / "shared-Q4" / "model.gguf").exists())

    def test_unknown_name(self):
        for bad in ("nosuch", "../data", "models/shared-Q4"):
            with self.assertRaises(SystemExit):
                momoe.cmd_remove([bad, "--yes"])
        self.assertTrue(self.outside.exists())

    def answers(self, *replies):
        it = iter(replies)
        return mock.patch.object(momoe.sys.stdin, "isatty", lambda: True), mock.patch("builtins.input", lambda _: next(it))

    def test_second_question_no_keeps_the_others(self):
        tty, inp = self.answers("y", "n")
        with tty, inp:
            self.assertEqual(momoe.cmd_remove(["b"]), 0)
        self.assertFalse(self.cfg_b.exists())
        self.assertTrue(self.cfg_c.exists() and (self.t / "models" / "shared-Q4").exists())

    def test_second_question_yes_takes_them(self):
        tty, inp = self.answers("y", "y")
        with tty, inp:
            self.assertEqual(momoe.cmd_remove(["b"]), 0)
        for gone in (self.cfg_b, self.cfg_c, self.t / "models" / "shared-Q4", self.t / "packs" / "shared-q4",
                     self.t / "logs" / "c.log"):
            self.assertFalse(gone.exists(), gone)
        self.assertTrue(self.cfg_a.exists() and self.outside.exists())

    def test_wipe_without_asking(self):
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: False):
            self.assertEqual(momoe.cmd_remove(["b", "--wipe"]), 0)
        self.assertFalse(self.cfg_c.exists() or (self.t / "models" / "shared-Q4").exists())
        self.assertTrue(self.cfg_a.exists())

    def test_yes_never_takes_the_others(self):
        self.assertEqual(momoe.cmd_remove(["b", "-y"]), 0)
        self.assertTrue(self.cfg_c.exists() and (self.t / "models" / "shared-Q4").exists())

    def test_wipe_follows_the_sharing(self):
        # d uses c's pack and a model of its own: wiping b takes c (the model), then d (the pack) and d's model
        m = self.t / "models" / "other-Q8"
        m.mkdir()
        (m / "model.gguf").write_bytes(b"x")
        d = self.t / "config" / "d.json"
        d.write_text(json.dumps({"exe": "x", "setup_tag": "other-Q8", "args": ["-m", str(m / "model.gguf")],
                                 "tokenizer": str(self.t / "packs" / "shared-q4" / "tokenizer")}))
        self.assertEqual(momoe.cmd_remove(["b", "--wipe"]), 0)
        self.assertFalse(d.exists() or m.exists())
        self.assertTrue(self.cfg_a.exists())

    def test_folder_in_use_asks_or_wipes(self):
        tty, inp = self.answers("n")
        with tty, inp:
            self.assertEqual(momoe.cmd_remove(["shared-Q4"]), 1)
        self.assertTrue(self.cfg_b.exists())
        self.assertEqual(momoe.cmd_remove(["shared-q4", "--wipe"]), 0)
        self.assertFalse(self.cfg_b.exists() or self.cfg_c.exists() or (self.t / "models" / "shared-Q4").exists())

    def test_wipe_refused_when_a_sharing_config_runs(self):
        with mock.patch.object(momoe, "running", lambda: {"pid": 1, "config": str(self.cfg_c), "model": "c"}), \
                self.assertRaises(SystemExit):
            momoe.cmd_remove(["b", "--wipe"])
        self.assertTrue(self.cfg_b.exists())


if __name__ == "__main__":
    unittest.main()
