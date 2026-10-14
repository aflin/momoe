"""tools/test_setup_qwen36_more.py - setup's other Qwen3.6-35B-A3B and KAT-Coder files (setup.MORE_SIZES,
docs/QWEN35MOE_GGUFS.md): their tables, the families qwen36-bartowski and kat-coder, that they are never suggested
over the regular sizes, their downloads and SHA-256, and the engine they need (FORMATS_ENGINE_MARK).  No GPU, no
download: the harness of test_setup_qwen36.py.

    python tools/test_setup_qwen36_more.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_setup_qwen36 as q36  # noqa: E402  (the qwen36 harness: setup imported, main() mocked)

setup = q36.setup
MORE = [m for m, d in setup.MODELS.items() if d.get("more")]


def family_sizes(family, more=None):
    return [m for m, d in setup.MODELS.items() if family in d.get("families", ())
            and (more is None or bool(d.get("more")) == more)]


class Tables(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(MORE), 36)
        self.assertEqual([len(family_sizes(f, True)) for f in ("qwen36", "qwen36-bartowski", "kat-coder")], [18, 16, 2])
        self.assertEqual(family_sizes("qwen36", False), [q36.IQ4, q36.IQ3])          # the regular two, unchanged
        self.assertNotIn("UD-Q8_K_XL", [setup.size_of(m) for m in MORE])             # does not load (BF16 attention)

    def test_families(self):
        self.assertEqual([f for f in list(setup.FAMILIES)[4:] if not setup.FAMILIES[f].get("ready")],
                         ["qwen36", "ornith", "qwen36-bartowski", "kat-coder"])   # (upstream's NVFP4 packs: not offered)
        tags = [d["tag"] for d in setup.FAMILIES.values() if d["tag"]]
        for t in tags:                                                                # config_family reads the prefix
            self.assertEqual([u for u in tags if u.startswith(t)], [t], t)
        for f in ("qwen36-bartowski", "kat-coder"):
            fam = setup.FAMILIES[f]
            self.assertTrue(setup.small_family(f))
            for k, v in {"own_mtp": True, "ple": False, "one_gpu": True, "shards": 1, "nvidia_only": True,
                         "mmproj": "mmproj-Qwen_Qwen3.6-35B-A3B-bf16.gguf", "profile": "expert-profile-qwen36.bin",
                         "pack_args": ["--compat-bf16"]}.items():
                self.assertEqual(fam[k], v, (f, k))
            self.assertIn("Apache 2.0", fam["license"])
        self.assertIn("/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/resolve/5c2410d71524f4f72b023ce8daf7a80528226d5f/",
                      setup.FAMILIES["qwen36-bartowski"]["hf"])
        self.assertIn("/Myric/KAT-Coder-V2.5-Dev-MTP-APEX-GGUF/resolve/e4ae9970325413d582ba2c294122fc5567ec1dd7/",
                      setup.FAMILIES["kat-coder"]["hf"])

    def test_files(self):
        for fam_name, size, nbytes, sha, experts, _tps in setup.MORE_SIZES:
            fam = setup.FAMILIES[fam_name]
            m = setup.model_key(fam_name, size)
            name = setup.model_file(fam, m, 1)
            self.assertEqual(fam["sha256"][name], (nbytes, sha), name)
            self.assertEqual(len(sha), 64)
            d = setup.MODELS[m]
            self.assertAlmostEqual(d["download_gb"], nbytes / 1e9, places=1)
            self.assertAlmostEqual(d["arena_gb"], experts / 1e9, places=1)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["engine"], setup.QWEN36_ENGINE)
        self.assertEqual(setup.model_file(setup.FAMILIES["kat-coder"], "KAT-Coder-i-quality-v2", 1),
                         "KAT-Coder-V2.5-Dev-MTP-APEX-i-quality-v2.gguf")
        self.assertEqual(setup.model_file(setup.FAMILIES["qwen36-bartowski"], "Qwen3.6-bartowski-Q4_1", 1),
                         "Qwen_Qwen3.6-35B-A3B-Q4_1.gguf")

    def test_ram_line_as_the_regular_sizes(self):
        # experts + ~11 GB, to an even number: what the measured lines of the regular sizes give
        for m in (q36.IQ4, q36.IQ3, "Ornith-1.5-IQ4_XS", "Ornith-1.5-IQ3_XXS"):
            d = setup.MODELS[m]
            self.assertEqual(2 * round((d["arena_gb"] + 11) / 2), d["ram_gb"], m)
        self.assertEqual(setup.MODELS["Qwen3.6-UD-Q4_K_XL"]["ram_gb"], 30)
        self.assertEqual(setup.MODELS["Qwen3.6-Q8_0"]["ram_gb"], 46)

    def test_model_keys(self):
        self.assertEqual(setup.model_key("qwen36", "UD-Q4_K_XL"), "Qwen3.6-UD-Q4_K_XL")
        self.assertEqual(setup.model_key("unsloth", "UD-Q4_K_XL"), "UD-Q4_K_XL")         # Flash-Next's stays
        self.assertEqual(setup.model_key("qwen36-bartowski", "IQ3_XXS"), "Qwen3.6-bartowski-IQ3_XXS")
        self.assertEqual(setup.model_key("ornith", "IQ3_XXS"), "Ornith-1.5-IQ3_XXS")
        self.assertEqual(setup.model_key("qwen", "IQ3_XXS"), "IQ3_XXS")
        self.assertEqual(setup.model_key("qwen36", "Q8_0"), "Qwen3.6-Q8_0")
        self.assertEqual(setup.model_key("qwen36-bartowski", "Q8_0"), "Qwen3.6-bartowski-Q8_0")
        self.assertIsNone(setup.model_key("qwen36", "Q4_1"))                              # bartowski's only
        self.assertEqual(setup.model_key("kat-coder", "I-QUALITY-V2"), "KAT-Coder-i-quality-v2")

    def test_never_suggested_over_the_regular_sizes(self):
        for ram in (q36.RAM16, q36.RAM24, q36.RAM32, q36.RAM64):
            self.assertIn(setup.small_size("qwen36", ram), (q36.IQ4, q36.IQ3))
            self.assertIn(setup.qwen36_size(ram), (q36.IQ4, q36.IQ3))
        self.assertEqual(setup.small_size("qwen36-bartowski", q36.RAM32), "Qwen3.6-bartowski-IQ4_XS")
        self.assertEqual(setup.small_size("qwen36-bartowski", q36.RAM24), "Qwen3.6-bartowski-IQ3_XXS")
        self.assertEqual(setup.small_size("kat-coder", q36.RAM64), "KAT-Coder-i-quality-v2")

    def test_gguf_names(self):
        self.assertIsNone(setup.gguf_unsupported("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))
        self.assertIsNone(setup.gguf_unsupported("Qwen_Qwen3.6-35B-A3B-Q4_1.gguf"))
        self.assertIsNone(setup.gguf_unsupported("KAT-Coder-V2.5-Dev-MTP-APEX-i-quality.gguf"))
        self.assertEqual(setup.gguf_unsupported("Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"), "UD-Q8_K_XL")
        # Flash-Next's Unsloth UD-IQ3_XXS (#444) is still not a file Strata runs
        self.assertEqual(setup.gguf_unsupported("Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf"), "UD-IQ3_XXS")
        self.assertEqual(setup.gguf_choice("Qwen_Qwen3.6-35B-A3B-Q6_K.gguf"), ("qwen36-bartowski", "Q6_K"))


class Install(q36.Base):
    def test_unsloth_ud_q4_k_xl(self):
        code, out, cfg = self.main(["--family", "qwen36", "--model", "UD-Q4_K_XL"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-q4_k_xl.json")
        native = q36.arg(cfg, "--native")
        self.assertTrue(native.endswith("models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), native)
        self.assertEqual(q36.arg(cfg, "--mtp"), native)
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-ud-q4_k_xl")
        self.assertEqual(self.downloads, [q36.BASE + "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"])
        self.assertIn(("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 22853663008,
                       "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"), self.verified)

    def test_bartowski_q4_1(self):
        code, out, cfg = self.main(["--family", "qwen36-bartowski", "--model", "Q4_1"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-bartowski-qwen36-q4_1.json")
        native = q36.arg(cfg, "--native")
        self.assertTrue(native.endswith("models/bartowski-qwen36-Q4_1/Qwen_Qwen3.6-35B-A3B-Q4_1.gguf"), native)
        self.assertEqual(q36.arg(cfg, "--mtp"), native)                   # bartowski's files carry the draft layer
        self.assertTrue(q36.arg(cfg, "--expert-profile").endswith("expert-profile-qwen36.bin"))
        self.assertNotIn("--ple-gguf", cfg["args"])
        self.assertEqual(cfg["model_name"], "qwen3.6-35b-a3b-bartowski-q4_1")
        self.assertTrue(all("bartowski/Qwen_Qwen3.6-35B-A3B-GGUF" in u for u in self.downloads), self.downloads)
        self.assertEqual(setup.config_family(self.cfg_path), "qwen36-bartowski")

    def test_kat_coder_default_size(self):
        code, out, cfg = self.main(["--family", "kat-coder"], ram=q36.RAM32)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-kat-coder-i-quality-v2.json")
        self.assertIn(("KAT-Coder-V2.5-Dev-MTP-APEX-i-quality-v2.gguf", 20666451296,
                       "f4501b4f54577da43a1d088bd3b81ce4df900a553f5407bc5d29bcbe54af66ef"), self.verified)
        self.assertEqual(setup.config_family(self.cfg_path), "kat-coder")

    def test_qwen36_default_stays(self):
        code, out, cfg = self.main(["--family", "qwen36"], ram=q36.RAM64)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.cfg_path.name, "strata-qwen36-ud-iq4_xs.json")
        self.assertIn("UD-Q4_K_XL", out)                                  # listed after the regular two
        self.assertLess(out.index("UD-IQ3_S"), out.index("UD-IQ1_M"))

    def test_a_strata_engine_is_compiled_over(self):
        # Momoe has no ready-made engine: one from Strata's releases is compiled over, these formats or not
        for m in ("MXFP4_MOE", "UD-IQ4_XS"):
            code, out, cfg = self.main(["--family", "qwen36", "--model", m], source="release", runs_more=False)
            self.assertEqual(code, 0, out)
            self.assertEqual(len(self.builds), 1, m)

    def test_older_checkout_stops_before_the_download(self):
        code, out, cfg = self.main(["--family", "qwen36-bartowski", "--model", "Q6_K"], runs_more=False)
        self.assertEqual(code, 1)
        self.assertIn("needs an engine with the expert formats added for it", out)
        self.assertEqual(self.downloads, [])

    def test_check_lists_them_in_one_line(self):
        code, out, cfg = self.main(["--check"])
        self.assertIn("36 more files of Qwen3.6-35B-A3B and KAT-Coder", out)
        self.assertNotIn("Qwen3.6-UD-Q4_K_XL", out)


class OnThisPc(unittest.TestCase):
    """The menus mark the sizes already here: set up (a config) or only downloaded."""

    def test_installed_downloaded_or_not(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            models = Path(d) / "models"
            self.assertEqual(setup.on_this_pc("qwen36", "Qwen3.6-UD-IQ1_M", models), "")
            (Path(d) / "config").mkdir()
            (Path(d) / "config" / "strata-qwen36-ud-iq1_m.json").write_text("{}")
            self.assertEqual(setup.on_this_pc("qwen36", "Qwen3.6-UD-IQ1_M", models), "installed")
            f = models / "bartowski-qwen36-Q4_1" / "Qwen_Qwen3.6-35B-A3B-Q4_1.gguf"
            f.parent.mkdir(parents=True)
            f.write_bytes(b"")
            self.assertEqual(setup.on_this_pc("qwen36-bartowski", "Qwen3.6-bartowski-Q4_1", models), "")  # unfinished
            setup.mark(f)
            self.assertEqual(setup.on_this_pc("qwen36-bartowski", "Qwen3.6-bartowski-Q4_1", models), "downloaded")
            self.assertEqual(setup.on_this_pc("qwen", "IQ3_XXS", models), "")


class Menus(q36.Base):
    def test_menus_show_what_is_here(self):
        here = {"Qwen3.6-UD-IQ1_M": "installed", "Qwen3.6-UD-Q4_K_XL": "downloaded"}
        with mock.patch.object(setup, "on_this_pc", lambda f, m, d: here.get(m, "")):
            code, out, cfg = self.main(["--setup"], answers={"Which model?": "5"})
        self.assertEqual(code, 0, out)
        family_line = next(ln for ln in out.splitlines() if ln.startswith("  5) "))
        self.assertIn("[on this PC: UD-IQ1_M installed, UD-Q4_K_XL downloaded]", family_line)
        self.assertNotIn("on this PC", next(ln for ln in out.splitlines() if ln.startswith("  1) ")))
        self.assertRegex(out, r"\) UD-IQ1_M .*\[installed\]")
        self.assertRegex(out, r"\) UD-Q4_K_XL .*\[downloaded\]")


if __name__ == "__main__":
    unittest.main()
