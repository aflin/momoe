"""tools/test_setup_momoe.py - Momoe's config names: `momoe setup` asks the config's file name (--config-name gives
it), a config named so carries "setup_tag", and setup and `momoe` find it by that.  No GPU, no download: the harness
of test_setup_qwen36.py.

    python tools/test_setup_momoe.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_setup_qwen36 as q36  # noqa: E402  (the qwen36 harness: setup imported, main() mocked)
import momoe  # noqa: E402

setup = q36.setup
ORNITH = ["--setup", "--family", "ornith", "--model", "Ornith-1.5-IQ4_XS"]


class Names(unittest.TestCase):
    def test_file_names(self):
        self.assertEqual(setup.config_file_name("ornith-8gb"), "ornith-8gb.json")
        self.assertEqual(setup.config_file_name(" mine.json "), "mine.json")
        for bad in ("", "a/b.json", "..\\x", "..", "x.shared-settings.json"):
            self.assertIsNone(setup.config_file_name(bad), bad)


class Setup(q36.Base):
    def written(self, name):
        p = self.t / "config" / name
        self.assertTrue(p.is_file(), sorted(x.name for x in self.t.glob("**/*.json")))
        return json.loads(p.read_text())

    def test_the_default_name_is_unchanged(self):
        code, out, cfg = self.main(ORNITH)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-ornith15-iq4_xs.json")
        self.assertNotIn("setup_tag", cfg)
        self.assertIn("config: strata-ornith15-iq4_xs.json", out)
        self.assertIn("momoe start strata-ornith15-iq4_xs.json", out)
        self.assertFalse(list(self.t.glob("run-*")))                     # no run scripts: momoe start

    def test_config_name(self):
        code, out, _ = self.main([*ORNITH, "--config-name", "ornith-8gb"])
        self.assertEqual(code, 0, out)
        cfg = self.written("ornith-8gb.json")
        self.assertEqual(cfg["setup_tag"], "ornith15-IQ4_XS")
        self.assertEqual(cfg["log"], str(self.t / "logs" / "ornith-8gb.log"))   # every log in logs/
        self.assertIn("momoe start ornith-8gb.json", out)
        p = self.t / "config" / "ornith-8gb.json"
        self.assertEqual(setup.config_tag(p), "ornith15-IQ4_XS")
        self.assertEqual(setup.config_family(p), "ornith")
        with mock.patch.object(setup, "ROOT", self.t):
            self.assertIn(p, setup.model_config_files(self.t))
            self.assertEqual(setup.tag_config("ornith15-IQ4_XS"), p)
            self.assertEqual(setup.on_this_pc("ornith", "Ornith-1.5-IQ4_XS", self.t / "data" / "models"), "installed")
            self.assertEqual(setup.choices_from_config(p)["model"], "Ornith-1.5-IQ4_XS")

    def test_asked(self):
        with mock.patch.object(setup.sys.stdin, "isatty", lambda: True), \
                mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            code, out, _ = self.main(ORNITH, answers={"Name for this config": "mine"})
        self.assertEqual(code, 0, out)
        self.assertTrue(any("Name for this config [strata-ornith15-iq4_xs.json]" in q for q in self.asked), self.asked)
        self.assertEqual(self.written("mine.json")["setup_tag"], "ornith15-IQ4_XS")

    def test_a_setup_again_offers_the_name_it_was_given(self):
        self.main([*ORNITH, "--config-name", "mine"])
        with mock.patch.object(setup.sys.stdin, "isatty", lambda: True), \
                mock.patch.object(setup, "flush_typed_ahead", lambda: None):
            code, out, _ = self.main(ORNITH, answers={})
        self.assertEqual(code, 0, out)
        self.assertTrue(any("Name for this config [mine.json]" in q for q in self.asked), self.asked)

    def test_other_json_is_no_config(self):
        (self.t / "package.json").write_text('{"name": "x"}')
        self.assertNotIn(self.t / "package.json", setup.model_config_files(self.t))


class EngineNames(unittest.TestCase):
    """The engine is momoe(.exe) and the encoder momoe-vision(.exe): an engine folder set up under Strata's names is
    renamed, and a config pointing at the old names follows."""

    def test_rename_and_config(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            old_exe, old_vexe = (eng / n for n in setup.STRATA_NAMES)
            old_exe.write_bytes(b"engine")
            old_vexe.write_bytes(b"encoder")
            setup.momoe_names(eng)
            self.assertEqual((eng / setup.EXE).read_bytes(), b"engine")
            self.assertEqual((eng / setup.VEXE).read_bytes(), b"encoder")
            self.assertFalse(old_exe.exists() or old_vexe.exists())
            setup.momoe_names(eng)                                       # again: nothing to do
            cfg_path = eng / "strata-x.json"
            cfg = {"exe": str(old_exe), "args": [], "vision": {"exe": str(old_vexe)}}
            cfg_path.write_text(json.dumps(cfg))
            with mock.patch.object(setup, "is_wsl", lambda: False):
                got = setup.upgrade_config(cfg_path, cfg)
            self.assertEqual(got["exe"], str(eng / setup.EXE))
            self.assertEqual(got["vision"]["exe"], str(eng / setup.VEXE))
            self.assertEqual(json.loads(cfg_path.read_text())["exe"], str(eng / setup.EXE))

    def test_a_momoe_engine_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d)
            old = eng / next(iter(setup.STRATA_NAMES))
            old.write_bytes(b"old")
            (eng / setup.EXE).write_bytes(b"new")
            setup.momoe_names(eng)
            self.assertEqual((eng / setup.EXE).read_bytes(), b"new")       # not replaced by the Strata-named one
            self.assertTrue(old.exists())


class DataFolder(unittest.TestCase):
    """Momoe keeps its model files in its own folder (models/, packs/, mtp/) and its settings apart from Strata's, so
    neither moves the other's files; Strata's calibrations carry over once."""

    def test_own_settings_and_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "strata").mkdir()
            (d / "strata" / "settings.json").write_text(json.dumps(
                {"data_dir": str(d / "Strata-data"), "installs": [str(d / "Strata")], "calibration": {"k": {"x": 1}}}))
            strata = d / "Strata"                       # a Strata folder next to this one, with model files
            (strata / "models" / "IQ2_XS").mkdir(parents=True)
            (strata / "models" / "IQ2_XS" / "a.gguf").write_bytes(b"x")
            (strata / "setup.py").write_text("")
            root = d / "momoe"
            root.mkdir()
            with mock.patch.dict(setup.os.environ, {"XDG_CONFIG_HOME": str(d)}), \
                    mock.patch.object(setup, "WIN", False), mock.patch.object(setup, "ROOT", root), \
                    mock.patch.object(setup, "say", lambda *a, **k: None), mock.patch.object(setup, "ok", lambda *a: None):
                self.assertEqual(setup.settings_path(), d / "momoe" / "settings.json")
                self.assertEqual(setup.load_settings(), {"calibration": {"k": {"x": 1}}})
                self.assertEqual(setup.data_folder(None), (root, []))
                saved = json.loads((d / "momoe" / "settings.json").read_text())
            self.assertNotIn("data_dir", saved)                          # the default is not remembered
            self.assertEqual(saved["installs"], [str(root)])
            self.assertTrue((strata / "models" / "IQ2_XS" / "a.gguf").exists())   # Strata's files stay put

    def test_models_folder_is_kept_empty_in_git(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "models" / ".gitkeep").is_file())
        lines = (root / ".gitignore").read_text().splitlines()
        self.assertIn("/models/*", lines)
        self.assertIn("!/models/.gitkeep", lines)


class Momoe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.ps = [mock.patch.object(momoe, "ROOT", self.t), mock.patch.object(momoe, "STATE_DIR", self.t / ".momoe"),
                   mock.patch.object(momoe, "STATE", self.t / ".momoe" / "server.json")]
        for p in self.ps:
            p.start()

    def tearDown(self):
        for p in self.ps:
            p.stop()
        self.tmp.cleanup()

    def three(self):
        cfg = {"exe": "x", "args": ["--max-context", "32768"]}
        (self.t / "config").mkdir()
        for n in ("a", "b", "c"):
            (self.t / "config" / f"{n}.json").write_text(json.dumps({**cfg, "setup_tag": n, "model_name": n}))
        (self.t / ".momoe").mkdir()
        (self.t / ".momoe" / "last.json").write_text(json.dumps({"config": str(self.t / "config" / "b.json")}))

    def test_run_config(self):
        self.three()
        b = (self.t / "config" / "b.json").resolve()
        self.assertEqual(momoe.run_config(["claude", "--config", "b", "-p", "hi"]), (b, ["claude", "-p", "hi"]))
        self.assertEqual(momoe.run_config(["--config=b.json", "unsloth"]), (b, ["unsloth"]))
        self.assertEqual(momoe.run_config(["--config", "b", "codex", "--config", "k=v"]),
                         (b, ["codex", "--config", "k=v"]))                  # the second one is Codex's own
        self.assertEqual(momoe.run_config(["codex", "-m", "x", "--config", "k=v"]),
                         (None, ["codex", "-m", "x", "--config", "k=v"]))
        with self.assertRaises(SystemExit):
            momoe.run_config(["claude", "--config", "missing"])

    def test_move_log(self):
        self.three()
        c = self.t / "config"
        old = self.t / "a.log"
        old.write_text("engine line\n")
        (c / "a.json").write_text(json.dumps({"exe": "x", "log": str(old)}))
        (c / "b.json").write_text(json.dumps({"exe": "x", "log": "/elsewhere/b.log"}))
        with mock.patch.object(momoe, "say"):
            momoe.move_log(c / "a.json")                                     # in the top level: moved
            momoe.move_log(c / "b.json")                                     # anywhere else: left alone
            momoe.move_log(c / "c.json")                                     # no log: nothing
        self.assertFalse(old.exists())
        self.assertEqual((self.t / "logs" / "a.log").read_text(), "engine line\n")
        self.assertEqual(json.loads((c / "a.json").read_text())["log"], str(self.t / "logs" / "a.log"))
        self.assertEqual(json.loads((c / "b.json").read_text())["log"], "/elsewhere/b.log")
        self.assertNotIn("log", json.loads((c / "c.json").read_text()))

    def test_log_follows(self):
        logs = self.t / "logs"
        logs.mkdir()
        server, engine = logs / "server.log", logs / "a.log"
        server.write_text("s1\ns2\n")
        engine.write_text("e1\n")
        (self.t / ".momoe").mkdir()
        (self.t / ".momoe" / "last.json").write_text(json.dumps({"config": str(self.t / "a.json")}))
        (self.t / "a.json").write_text(json.dumps({"exe": "x", "log": str(engine)}))
        out, ticks = [], []

        def tick(_):                       # each poll: the logs grow, then the second poll ends it
            ticks.append(1)
            if len(ticks) == 1:
                with open(engine, "a") as f:
                    f.write("e2\npart")
                server.write_text("new\n")             # shorter: the server started again
            elif len(ticks) == 2:
                with open(engine, "a") as f:
                    f.write("ial\n")
            else:
                raise KeyboardInterrupt
        with mock.patch.object(momoe, "say", out.append), mock.patch.object(momoe.time, "sleep", tick):
            self.assertEqual(momoe.cmd_log(["-n", "1"]), 0)
        lines = [x for x in out if not x.startswith(("==>", "---"))]
        self.assertEqual(lines, ["[server] s2", "[engine] e1", "[server] new", "[engine] e2", "[engine] partial"])
        out.clear()
        with mock.patch.object(momoe, "say", out.append):
            self.assertEqual(momoe.cmd_log(["engine", "--no-follow"]), 0)
        self.assertEqual(out[1:], ["e1", "e2", "partial"])          # one log: no tags

    def test_pick_config(self):
        self.three()
        c = self.t / "config"
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: False):
            self.assertEqual(momoe.pick_config(), c / "b.json")              # no terminal: the one started last
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: True), mock.patch.object(momoe, "say"), \
                mock.patch("builtins.input", side_effect=["", "9", "2", "c"]):
            self.assertEqual(momoe.pick_config(), c / "b.json")              # Enter: the one started last
            self.assertIn(momoe.pick_config(), (c / "a.json", c / "c.json"))  # 9: asked again, then 2
            self.assertEqual(momoe.pick_config(), c / "c.json")              # by name
        for n in ("a", "c"):
            (c / f"{n}.json").unlink()
        with mock.patch.object(momoe.sys.stdin, "isatty", lambda: True), \
                mock.patch("builtins.input", side_effect=AssertionError("asked")):
            self.assertEqual(momoe.pick_config(), c / "b.json")              # only one: not asked

    def test_a_running_server_is_not_switched(self):
        self.three()
        c = self.t / "config"
        st = {"pid": 1, "config": str(c / "a.json"), "model": "a"}
        with mock.patch.object(momoe, "running", lambda: st), mock.patch.object(momoe, "health", lambda s: {}), \
                mock.patch.object(momoe, "start_config", mock.Mock(side_effect=AssertionError("started"))):
            self.assertEqual(momoe.ensure_server(), (st, {}))               # no --config: the running one
            self.assertEqual(momoe.ensure_server(c / "a.json"), (st, {}))
            with self.assertRaises(SystemExit):
                momoe.ensure_server(c / "b.json")

    def test_configs(self):
        cfg = {"exe": "strata", "args": ["--max-context", "32768"]}
        (self.t / "strata-q2_0.json").write_text(json.dumps(cfg))        # from before config/: moved there
        (self.t / "strata-q2_0.shared-settings.json").write_text("{}")   # with its Chat settings
        (self.t / "config").mkdir()
        (self.t / "config" / "mine.json").write_text(json.dumps({**cfg, "setup_tag": "ornith15-IQ4_XS"}))
        (self.t / "config" / "other.json").write_text(json.dumps(cfg))   # no setup_tag: not one of setup's
        self.assertEqual(sorted(p.name for p in momoe.configs()), ["mine.json", "strata-q2_0.json"])
        self.assertTrue((self.t / "config" / "strata-q2_0.shared-settings.json").is_file())
        self.assertFalse(list(self.t.glob("*.json")))
        self.assertEqual(momoe.find_config("mine").name, "mine.json")
        self.assertEqual(momoe.find_config("q2_0").name, "strata-q2_0.json")
        with self.assertRaises(SystemExit):
            momoe.find_config("missing")

    def test_lmstudio_plugin_settings(self):
        home = self.t / "lmstudio"
        cfgf = home / ".internal" / "global-plugin-configs.json"
        with mock.patch.object(momoe, "lmstudio_home", lambda: home):
            self.assertTrue(momoe.lmstudio_plugin_config("http://127.0.0.1:8090/v1", None))
            self.assertEqual(json.loads(cfgf.read_text()), {   # the layout LM Studio itself read (2026-10-12)
                "json": {"plugins": [[momoe.LMS_PLUGIN, {"fields": [
                    {"key": "overrideBaseUrl", "value": "http://127.0.0.1:8090/v1"}]}]]},
                "meta": {"values": {"plugins": ["map"]}}})
            self.assertFalse(momoe.lmstudio_plugin_config("http://127.0.0.1:8090/v1", None))   # already so
            d = json.loads(cfgf.read_text())                 # another plugin's settings, and one of this plugin's own
            d["json"]["plugins"].insert(0, ["x/other", {"fields": [{"key": "a", "value": 1}]}])
            d["json"]["plugins"][1][1]["fields"].append({"key": "extra", "value": "kept"})
            cfgf.write_text(json.dumps(d))
            self.assertTrue(momoe.lmstudio_plugin_config("http://127.0.0.1:8080/v1", "k1"))  # another port, a key
            got = dict(json.loads(cfgf.read_text())["json"]["plugins"])
            self.assertEqual(got["x/other"], {"fields": [{"key": "a", "value": 1}]})
            self.assertEqual({f["key"]: f["value"] for f in got[momoe.LMS_PLUGIN]["fields"]},
                             {"overrideBaseUrl": "http://127.0.0.1:8080/v1", "openaiApiKey": "k1", "extra": "kept"})
            self.assertTrue(momoe.lmstudio_model_list("m1"))
            self.assertFalse(momoe.lmstudio_model_list("m1"))
            self.assertTrue(momoe.lmstudio_model_list("m2"))
            cache = json.loads((home / "openai-compat-endpoint" / "models-cache.json").read_text())
            self.assertEqual(cache, {"models": ["m2", "m1"], "lastSelected": "m1"})

    def test_lmstudio_prepare_order(self):
        home = self.t / "lmstudio"
        cfgf = home / ".internal" / "global-plugin-configs.json"
        seen = []

        def fake_run(cmd, **kw):                             # lms get starts LM Studio: the settings must be there
            seen.append((cmd[1:], cfgf.is_file()))
            m = home / "extensions" / "plugins" / momoe.LMS_PLUGIN / "manifest.json"
            m.parent.mkdir(parents=True)
            m.write_text("{}")
            return mock.Mock(returncode=0)
        s = {"port": 8090, "config": str(self.t / "c.json")}
        (self.t / "c.json").write_text("{}")
        with mock.patch.object(momoe, "lmstudio_home", lambda: home), mock.patch.object(momoe, "say"), \
                mock.patch.object(momoe, "lmstudio_running", lambda: False), \
                mock.patch.object(momoe.subprocess, "run", fake_run):
            self.assertTrue(momoe.lmstudio_prepare(s, {"model": "m1"}, "/x/lms"))
            self.assertEqual(seen, [(["get", momoe.LMS_PLUGIN, "-y"], True)])
            self.assertTrue(momoe.lmstudio_prepare(s, {"model": "m1"}, "/x/lms"))   # all there: nothing again
            self.assertEqual(len(seen), 1)
        with mock.patch.object(momoe, "lmstudio_home", lambda: home), mock.patch.object(momoe, "say"), \
                mock.patch.object(momoe, "lmstudio_running", lambda: True):
            self.assertTrue(momoe.lmstudio_prepare(s, {"model": "m1"}, "/x/lms"))   # running, nothing to write
            self.assertFalse(momoe.lmstudio_prepare({**s, "port": 8080}, {"model": "m1"}, "/x/lms"))  # restart

    def test_server_key(self):
        c = self.t / "c.json"
        c.write_text(json.dumps({"api_key": "from-config"}))
        self.assertEqual(momoe.server_key({"config": str(c)}), "from-config")
        self.assertEqual(momoe.server_key({"config": str(c), "api_key": "once,two"}), "once")   # this start's own

    def test_base_url_and_key(self):
        self.assertEqual(momoe.base_url({"host": "0.0.0.0", "port": 8080}), "http://127.0.0.1:8080")
        self.assertEqual(momoe.base_url({"port": 8090}), "http://127.0.0.1:8090")
        c = self.t / "c.json"
        c.write_text(json.dumps({"api_key": "k1, k2"}))
        self.assertEqual(momoe.api_key(c), "k1")
        c.write_text("{}")
        self.assertIsNone(momoe.api_key(c))

    def test_a_missing_tool_is_named_not_installed(self):
        with mock.patch.object(momoe, "which", lambda n: None), \
                mock.patch.object(momoe, "lmstudio_paths", lambda: (None, None)), \
                mock.patch.object(momoe, "ensure_server", mock.Mock(side_effect=AssertionError("started"))), \
                mock.patch.object(momoe.subprocess, "Popen", mock.Mock(side_effect=AssertionError("ran"))), \
                mock.patch.object(momoe.subprocess, "call", mock.Mock(side_effect=AssertionError("ran"))), \
                mock.patch.object(momoe, "say") as said:
            for tool in momoe.TOOLS:
                self.assertEqual(momoe.cmd_run([tool]), 2, tool)
        text = "\n".join(str(c.args[0]) for c in said.call_args_list if c.args)
        for words in ("npm install -g @openai/codex", "https://unsloth.ai/install.sh", "https://lmstudio.ai/download"):
            self.assertIn(words, text)


if __name__ == "__main__":
    unittest.main()
