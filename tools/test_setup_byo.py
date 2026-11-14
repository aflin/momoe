"""tools/test_setup_byo.py - setup --gguf, a GGUF that is not in setup's list (a file of your own, or one on Hugging Face):
its header read and checked before anything is downloaded, a family and a size made for it from the nearest listed one,
and a gated repository's token sent to huggingface.co only.  No network, no GPU: the files are small GGUFs made here."""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key, t, v) -> bytes:
    out = s(key) + struct.pack("<I", t)
    if t == 8:
        return out + s(v)
    if t == 9:                                         # (type, values)
        at, vals = v
        out += struct.pack("<IQ", at, len(vals))
        return out + b"".join(s(x) if at == 8 else struct.pack("<I", x) for x in vals)
    return out + struct.pack({4: "<I", 5: "<i", 10: "<Q", 7: "<?"}[t], v)


def gguf(path: Path, keys: list, tensors: list, data_bytes: int = 0) -> Path:
    """A GGUF with these keys and tensors [(name, ggml type, n elements, offset)], and `data_bytes` of data after it."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keys)) + b"".join(kv(*k) for k in keys)
    for name, t, n, off in tensors:
        head += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<IQ", t, off)
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + b"\1" * data_bytes)
    return path


def experts(layers: int, gu: int, d: int, size: int = 100) -> list:
    out, off = [], 0
    for layer in range(layers):
        for role, t in (("gate", gu), ("up", gu), ("down", d)):
            out.append((f"blk.{layer}.ffn_{role}_exps.weight", t, 256, off))
            off += size
    return out


class Header(unittest.TestCase):
    def test_keys_tensors_and_the_data_start(self):
        with tempfile.TemporaryDirectory() as d:
            p = gguf(Path(d) / "m.gguf", [("general.architecture", 8, "qwen4exp"), ("split.count", 4, 1),
                                         ("tokenizer.ggml.tokens", 9, (8, ["a", "bb", "ccc"])),
                                         ("x.ids", 9, (4, [1, 2, 3])), ("qwen4exp.nextn_predict_layers", 4, 1)],
                     [("token_embd.weight", 8, 512, 0), ("blk.0.ffn_down_exps.weight", 23, 256, 544)], 1000)
            h = setup.GgufReader(p).header()
        self.assertEqual(h["kv"]["general.architecture"], "qwen4exp")
        self.assertEqual(h["kv"]["tokenizer.ggml.tokens"], 3)          # (an array: its length)
        self.assertEqual([t[0] for t in h["tensors"]], ["token_embd.weight", "blk.0.ffn_down_exps.weight"])
        self.assertEqual(h["data_start"] % 32, 0)

    def test_not_a_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.gguf"
            p.write_bytes(b"PK\3\4 not a gguf")
            with self.assertRaisesRegex(ValueError, "not a GGUF"):
                setup.GgufReader(p).header()

    def test_a_header_larger_than_the_first_read(self):
        with tempfile.TemporaryDirectory() as d:
            many = [(f"k{i}", 8, "v" * 2000) for i in range(800)]      # ~1.6 MB of keys: more than the first 1 MiB read
            p = gguf(Path(d) / "big.gguf", [("general.architecture", 8, "gemma4"), *many], [("t", 0, 4, 0)], 16)
            self.assertEqual(setup.GgufReader(p).header()["kv"]["k799"], "v" * 2000)


class Check(unittest.TestCase):
    def heads(self, d, arch="qwen4exp", gu=18, dn=18, parts=2, ple_part=1, mtp=0):
        hs, names, sizes = [], [], []
        for i in range(parts):
            keys = [("general.architecture", 8, arch), ("split.no", 2 if False else 4, i), ("split.count", 4, parts),
                    (f"{arch}.nextn_predict_layers", 4, mtp)]
            ts = experts(2, gu, dn) if i == parts - 1 else [("token_embd.weight", 8, 512, 0)]
            if i == ple_part:
                ts = ts + [("per_layer_token_embd.weight", 8, 64, 600 if i == parts - 1 else 544)]
            p = gguf(Path(d) / f"M-{i + 1:05d}-of-{parts:05d}.gguf", keys, ts, 20_000)
            hs.append(setup.GgufReader(p).header())
            names.append(p.name)
            sizes.append(p.stat().st_size)
        return hs, names, sizes

    def test_a_flash_next_file(self):
        with tempfile.TemporaryDirectory() as d:
            c = setup.byo_check(*self.heads(d), strata=True)
        self.assertEqual((c["arch"], c["ple"], c["mtp"]), ("qwen4exp", 1, False))
        self.assertEqual(c["expert_bytes"], 600)                        # 2 layers x 3 projections x 100 bytes

    def test_down_projections_strata_has_no_kernels_for(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                setup.byo_check(*self.heads(d, gu=16, dn=16), strata=True)       # IQ2_XXS down: no kernel
        self.assertIn("down IQ2_XXS (layers 0, 1)", f.call_args[0][0])
        self.assertNotIn("gate IQ2_XXS", f.call_args[0][0])                     # (IQ2_XXS gate/up has one)

    def test_the_ggml_engine_takes_any_format(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(setup.byo_check(*self.heads(d, arch="gemma4", gu=16, dn=16, parts=1, ple_part=None),
                                             strata=False)["arch"], "gemma4")

    def test_an_architecture_no_engine_runs(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                setup.byo_check(*self.heads(d, arch="llama", parts=1), strata=False)
        self.assertIn("is a llama model", f.call_args[0][0])

    def test_parts_that_do_not_belong_together(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            hs, names, sizes = self.heads(d)
            hs[1]["kv"]["split.no"] = 0
            with self.assertRaises(SystemExit):
                setup.byo_check(hs, names, sizes, strata=True)
        self.assertIn("is not part 2 of 2", f.call_args[0][0])

    def test_setups_format_lists_are_the_engines(self):
        src = (ROOT / "src" / "kernels" / "cuda" / "iq_kernels.cu").read_text(encoding="utf-8")
        gu = re.search(r"#define STRATA_GU_FMTS\(X\)(.*)", src).group(1)      # (the first: the default Q6_K build's)
        dn = re.search(r"#define STRATA_D_FMTS\(X\)(.*)", src).group(1)
        self.assertEqual(setup.BYO_GU_TYPES, {int(x) for x in re.findall(r"X\((\d+)\)", gu)})
        self.assertEqual(setup.BYO_D_TYPES, {int(x) for x in re.findall(r"X\((\d+)\)", dn)})


class Labels(unittest.TestCase):
    ORCA = [f"Qwen3.8-Flash-Next-Uncensored-{q}-{i:05d}-of-{n:05d}.gguf"
            for q, n in (("IQ3_XXS", 2), ("IQ4_XS", 3), ("IQ4_XS-MTP", 3), ("Q8_0", 5)) for i in range(1, n + 1)] + \
        ["Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf", "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf", "README.md"]

    def test_a_repository_of_split_models(self):
        sets = setup.byo_sets(self.ORCA)
        self.assertEqual(sorted(sets), ["IQ3_XXS", "IQ4_XS", "IQ4_XS-MTP", "Q8_0"])
        self.assertEqual(len(sets["Q8_0"]), 5)
        self.assertEqual(setup.byo_pick(sets, "iq3_xxs", "hf:x/y"), "IQ3_XXS")

    def test_a_draft_model_named_like_a_model_is_called_by_its_name(self):
        sets = setup.byo_sets(["g/gemma-4-it-Q8_0.gguf", "g/mtp-gemma-4-it-Q8_0.gguf", "g/gemma-4-it-UD-IQ3_XXS.gguf"])
        self.assertEqual(sorted(sets), ["UD-IQ3_XXS", "gemma-4-it-Q8_0", "mtp-gemma-4-it-Q8_0"])
        self.assertEqual(setup.byo_pick(sets, "gemma-4-it-Q8_0", "hf:x/y"), "gemma-4-it-Q8_0")

    def test_no_choice_among_several_lists_them(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                setup.byo_pick(setup.byo_sets(self.ORCA), None, "hf:orcarouter/x")
        self.assertIn("IQ3_XXS", f.call_args[0][1])
        self.assertIn("--gguf hf:orcarouter/x:", f.call_args[0][1])


class Token(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")}
        env["HF_HOME"] = self.tmp.name
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(setup, "_hf_token", None)
        q.start()
        self.addCleanup(q.stop)

    def test_where_the_token_is_found(self):
        self.assertIsNone(setup.hf_token())
        (Path(self.tmp.name) / "token").write_text("hf_file\n")
        self.assertEqual(setup.hf_token(), "hf_file")                  # the token `hf auth login` keeps
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_env"}):
            self.assertEqual(setup.hf_token(), "hf_env")
            own = Path(self.tmp.name) / "mine"
            own.write_text("hf_own")
            self.assertEqual(setup.hf_token(str(own)), "hf_own")        # --hf-token-file first

    def test_the_token_goes_to_huggingface_co_only(self):
        with mock.patch.object(setup, "_hf_token", "hf_secret"):
            r, opener = setup.hf_auth(urllib.request.Request("https://huggingface.co/x/y/resolve/abc/f.gguf"))
            self.assertEqual(r.get_header("Authorization"), "Bearer hf_secret")
            self.assertIsNotNone(opener)
            for other in ("https://cas-bridge.xethub.hf.co/x", "https://modelscope.cn/x", "https://huggingface.co.evil.com/"):
                r, opener = setup.hf_auth(urllib.request.Request(other))
                self.assertIsNone(r.get_header("Authorization"))
                self.assertIsNone(opener)

    def test_a_redirect_to_the_storage_host_drops_it(self):
        h = setup._DropAuthElsewhere()
        req = urllib.request.Request("https://huggingface.co/x/y/resolve/abc/f.gguf", headers={"Authorization": "Bearer t"})
        away = h.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/signed?x=1")
        self.assertIsNone(away.get_header("Authorization"))
        same = h.redirect_request(req, None, 307, "Temporary", {}, "https://huggingface.co/x/y/resolve/abc/g.gguf")
        self.assertEqual(same.get_header("Authorization"), "Bearer t")

    def test_a_gated_repository_without_a_token(self):
        a = types.SimpleNamespace(yes=True, hf_token_file=None)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                setup.byo_token_for("o/r", {"gated": "auto"}, a)
        self.assertIn("needs a Hugging Face token", f.call_args[0][0])
        self.assertIn("https://huggingface.co/o/r", f.call_args[0][1])
        setup.byo_token_for("o/open", {"gated": False}, a)              # an open one needs none
        self.assertIsNone(setup._hf_token)

    def test_a_typed_token_is_kept_only_when_asked(self):
        a = types.SimpleNamespace(yes=False, hf_token_file=None)
        with mock.patch.object(setup.sys.stdin, "isatty", return_value=True), \
                mock.patch("getpass.getpass", return_value="hf_typed"), mock.patch.object(setup, "ask", return_value="n"), \
                mock.patch.object(setup, "say"):
            setup.byo_token_for("o/r", {"gated": "manual"}, a)
        self.assertEqual(setup._hf_token, "hf_typed")
        self.assertFalse((Path(self.tmp.name) / "token").exists())


class Prepare(unittest.TestCase):
    """byo_prepare: the family and the size it makes, from a resolved model (byo_resolve mocked)."""

    def setUp(self):
        for p in (mock.patch.dict(setup.FAMILIES), mock.patch.dict(setup.MODELS), mock.patch.dict(setup.GGML_FAMILIES),
                  mock.patch.object(setup, "_byo_record", None), mock.patch.object(setup, "say")):
            p.start()
            self.addCleanup(p.stop)

    def resolved(self, arch, parts, local=False, mmproj=True, mtp=0, draft=False):
        base = "https://huggingface.co/o/R-GGUF/resolve/c0ffee/"
        rows = [((None if local else base + n), n, 1000 * (i + 1), (None if local else f"{i:064x}"),
                 (Path("/models/R") / n if local else None)) for i, n in enumerate(parts)]
        heads = [{"kv": {"general.architecture": arch, "split.count": len(parts), "split.no": i,
                         f"{arch}.nextn_predict_layers": mtp}, "tensors": [], "data_start": 64} for i in range(len(parts))]
        return {"where": "/models/R" if local else "hf:o/R-GGUF", "label": "IQ3_XXS", "parts": rows, "heads": heads,
                "mmproj": (base + "mmproj-F16.gguf", "mmproj-F16.gguf", 900, "ab" * 32) if mmproj else None,
                "draft": (base + "R-MTP-draft.gguf", "R-MTP-draft.gguf", 4_000, "cd" * 32, None) if draft else None,
                "repo": None if local else "o/R-GGUF", "commit": None if local else "c0ffee"}

    def run_prepare(self, r, strata=True, mtp=False):
        a = types.SimpleNamespace(gguf="x", family=None, model=None, gguf_dir=None)
        with mock.patch.object(setup, "byo_resolve", return_value=r), \
                mock.patch.object(setup, "byo_strata_path", return_value=strata), \
                mock.patch.object(setup, "byo_check", return_value={"arch": r["heads"][0]["kv"]["general.architecture"],
                                                                    "name": "R", "expert_bytes": 40e9, "ple": 1,
                                                                    "mtp": mtp}):
            setup.byo_prepare(a)
        return a

    def test_flash_next_on_strata_engine(self):
        names = [f"R-IQ3_XXS-{i:05d}-of-00002.gguf" for i in (1, 2)]
        a = self.run_prepare(self.resolved("qwen4exp", names))
        self.assertEqual((a.family, a.model), (setup.BYO, "R-GGUF IQ3_XXS"))
        fam, m = setup.FAMILIES[setup.BYO], setup.MODELS["R-GGUF IQ3_XXS"]
        self.assertEqual([setup.model_file(fam, a.model, i) for i in (1, 2)], names)   # the published names
        self.assertEqual(setup.model_shards(fam, a.model), 2)
        self.assertEqual(fam["hf"], "https://huggingface.co/o/R-GGUF/resolve/c0ffee/")
        self.assertEqual(fam["sha256"][names[1]], (2000, f"{1:064x}"))
        self.assertEqual((m["arena_gb"], m["budget"], m["families"]), (40.0, True, [setup.BYO]))
        self.assertEqual((fam["mmproj"], fam["vision"]), ("mmproj-F16.gguf", True))
        self.assertEqual(setup._byo_record["base"], "unsloth")
        self.assertIsNone(a.gguf_dir)

    def test_the_repositorys_own_draft_head(self):
        a = self.run_prepare(self.resolved("qwen4exp", ["R-IQ3_XXS.gguf"], draft=True))
        fam = setup.FAMILIES[setup.BYO]
        self.assertEqual(fam["mtp_gguf"][1], "R-MTP-draft.gguf")
        self.assertEqual(fam["sha256"]["R-MTP-draft.gguf"], (4_000, "cd" * 32))
        self.assertEqual(setup._byo_record["mtp_gguf"], "R-MTP-draft.gguf")
        a = self.run_prepare(self.resolved("qwen4exp", ["R-IQ3_XXS.gguf"]))
        self.assertNotIn("mtp_gguf", setup.FAMILIES[setup.BYO])

    def test_flash_next_with_the_head_inside_is_refused(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                self.run_prepare(self.resolved("qwen4exp", ["R-IQ4_XS-MTP.gguf"], draft=True), mtp=True)
        self.assertIn("MTP draft head inside", f.call_args[0][0])
        self.assertIn("R-MTP-draft.gguf", f.call_args[0][1])

    def test_a_local_file_uses_its_folder(self):
        a = self.run_prepare(self.resolved("qwen4exp", ["R-IQ3_XXS.gguf"], local=True, mmproj=False))
        self.assertEqual(a.gguf_dir, "/models/R")
        fam = setup.FAMILIES[setup.BYO]
        self.assertEqual(fam["sha256"]["R-IQ3_XXS.gguf"], (1000, None))
        # no picture reader beside it: the base model's, pinned (a fine-tune keeps Flash-Next's vision tower)
        self.assertEqual((fam["mmproj"], fam["mmproj_hf"]), (setup.FAMILIES["unsloth"]["mmproj"],
                                                             setup.FAMILIES["unsloth"]["mmproj_hf"]))

    def test_a_ggml_family(self):
        a = self.run_prepare(self.resolved("glm5-next", [f"Q/G-UD-IQ2_XXS-{i:05d}-of-00002.gguf" for i in (1, 2)]),
                             strata=False)
        fam = setup.GGML_FAMILIES[setup.BYO]
        self.assertEqual((a.family, a.model), (setup.BYO, "IQ3_XXS"))
        self.assertEqual([p[0] for p in fam["parts"]["IQ3_XXS"]], ["Q/G-UD-IQ2_XXS-00001-of-00002.gguf",
                                                                   "Q/G-UD-IQ2_XXS-00002-of-00002.gguf"])
        self.assertEqual(setup.ggml_parts(fam, "IQ3_XXS")[1][1], "G-UD-IQ2_XXS-00002-of-00002.gguf")
        self.assertEqual(fam["min_vram_gb"], setup.GGML_FAMILIES["glm53"]["min_vram_gb"])   # (the base's settings)
        self.assertEqual(setup.momoe_mmproj(setup.BYO, fam)[1], "mmproj-F16.gguf")

    def test_family_or_model_beside_it(self):
        a = types.SimpleNamespace(gguf="x", family="qwen", model=None)
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                setup.byo_prepare(a)

    def test_a_qwen_model_on_the_ggml_engine_is_not_yet(self):
        with mock.patch.object(setup, "fail", side_effect=SystemExit(2)) as f:
            with self.assertRaises(SystemExit):
                self.run_prepare(self.resolved("qwen4exp", ["R.gguf"]), strata=False)
        self.assertIn("runs on Strata's engine", f.call_args[0][0])


class DraftHead(unittest.TestCase):
    """The repository's own MTP draft head (orcarouter's ...-MTP-draft.gguf): found beside the model, and made into the
    draft layer unless it is the original's head or not one tools/mtp_from_gguf.py reads."""

    def test_which_files_are_draft_heads(self):
        files = ["Q-IQ4_XS-MTP-00001-of-00003.gguf", "Q-MTP-draft.gguf", "mmproj-Q-F16.gguf", "Q-IQ3_XXS-00001-of-00002.gguf",
                 "sub/q-mtp-draft-q8_0.gguf", "README.md"]
        self.assertEqual(setup.byo_draft_files(files), ["Q-MTP-draft.gguf", "sub/q-mtp-draft-q8_0.gguf"])
        self.assertNotIn("Q-MTP-draft", setup.byo_sets(files))          # not a model of its own

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.d, True)
        self.pack, self.mtp = self.d / "packs" / "byo-r", self.d / "mtp"
        (self.d / "models").mkdir()
        self.mtp.mkdir()
        (self.mtp / "mtp-manifest.json").write_text("[]")
        self.fam = {"mtp_gguf": ("https://huggingface.co/o/R/resolve/c/R-MTP-draft.gguf", "R-MTP-draft.gguf", 10, "ab" * 32,
                                 None)}
        self.calls = []
        for p in (mock.patch.object(setup, "say"), mock.patch.object(setup, "warn"),
                  mock.patch.object(setup, "mtp_corrupt", return_value=False),
                  mock.patch.object(setup, "download", side_effect=lambda *a, **k: self.calls.append(("download", a[1].name)))):
            p.start()
            self.addCleanup(p.stop)

    def layer(self, rc=0, same=None):
        own = self.pack / "mtp-own"

        def run(cmd, **k):
            tool = Path(str(cmd[1])).name
            self.calls.append((tool, "--ref" in cmd))
            if tool == "mtp_from_gguf.py":
                (own / "tensors").mkdir(parents=True, exist_ok=True)
                if same is not None:
                    (own / "mtp-compare.json").write_text(json.dumps({"same": same}))
            if tool == "mtp_rt.py":
                (own / "rt").mkdir(parents=True, exist_ok=True)
                (own / "rt" / "experts.bin").write_bytes(b"x")
            return types.SimpleNamespace(returncode=rc if tool == "mtp_from_gguf.py" else 0, stdout="")
        with mock.patch.object(setup, "run", side_effect=run):
            return setup.byo_draft_layer(self.fam, None, [self.d / "models" / "R-1.gguf"], self.pack, self.mtp, {})

    def test_a_fine_tunes_head_becomes_the_draft_layer(self):
        rt = self.layer(same=False)
        self.assertEqual(rt, self.pack / "mtp-own" / "rt")
        self.assertEqual([c[0] for c in self.calls], ["download", "mtp_from_gguf.py", "mtp_pack.py", "mtp_rt.py"])
        self.assertTrue(self.calls[1][1])                                # compared with the original's tensors
        self.assertFalse((self.pack / "mtp-own" / "tensors").exists())   # the 5 GB of tensors are not kept
        self.calls.clear()
        self.assertEqual(self.layer(), rt)                               # done: nothing again
        self.assertEqual(self.calls, [])

    def test_the_original_head_uses_the_original_draft_layer(self):
        self.assertIsNone(self.layer(same=True))
        self.assertNotIn("mtp_pack.py", [c[0] for c in self.calls])
        self.calls.clear()
        self.assertIsNone(self.layer())                                  # decided: not converted again
        self.assertEqual(self.calls, [])

    def test_a_head_it_cannot_read_falls_back(self):
        for rc in (3, 4):
            self.calls.clear()
            self.assertIsNone(self.layer(rc=rc))
            self.assertNotIn("mtp_pack.py", [c[0] for c in self.calls])
            setup.warn.assert_called()

    def test_without_the_originals_tensors_nothing_is_compared(self):
        (self.mtp / "mtp-manifest.json").unlink()
        self.layer()
        self.assertFalse(self.calls[1][1])

    def test_a_local_draft_head_is_not_downloaded(self):
        self.fam["mtp_gguf"] = (None, "R-MTP-draft.gguf", 10, None, self.d / "R-MTP-draft.gguf")
        self.layer()
        self.assertNotIn("download", [c[0] for c in self.calls])


class Kept(unittest.TestCase):
    def test_a_local_file_without_a_published_hash_is_not_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            p.write_bytes(b"x" * 10)
            with mock.patch.object(setup, "_hash_file") as h:
                setup.verify_sha256(p, 10, None, own=True)
            h.assert_not_called()

    def test_the_config_records_the_source_and_the_family(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "x.json"
            rec = {"source": "hf:o/r", "label": "IQ3_XXS", "arch": "qwen35moe", "base": "qwen36"}
            with mock.patch.object(setup, "_byo_record", rec), mock.patch.object(setup, "say"), \
                    mock.patch.object(setup, "ok"):
                setup.write_setup_config(cfg, {"exe": "e", "args": [], "setup_tag": "byo-r-iq3_xxs-iq3_xxs"})
            self.assertEqual(json.loads(cfg.read_text())["byo"], rec)
            self.assertEqual(setup.config_family(cfg), "qwen36")


if __name__ == "__main__":
    unittest.main()
