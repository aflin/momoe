"""tools/mtp_from_gguf.py - the MTP draft layer's tensors from a llama.cpp GGUF that carries Flash-Next's MTP head.

llama.cpp's qwen4exp converter (ggml-org/llama.cpp PR #27836, `--mtp`) writes the checkpoint's `mtp.*` tensors as one
trailing block `blk.N` (N = the trunk's layer count) with the trunk's tensor names, a standalone draft GGUF (`...-MTP-
draft.gguf`, `mtp-...-Q8_0.gguf`) or inside a model file.  This turns them back into the checkpoint's 31 tensors, as
tools/mtp_fetch.py writes them (one BF16 file each plus mtp-manifest.json), so tools/mtp_pack.py and tools/mtp_rt.py
make the draft layer from a fine-tune's own MTP head exactly as they make it from the original checkpoint's:

    python tools/mtp_from_gguf.py --gguf Qwen3.8-Flash-Next-Uncensored-MTP-draft.gguf --out DIR [--ref <Strata>/mtp]

What the converter changed, undone here (conversion/qwen.py, conversion/qwen4exp.py of that PR):
  every RMSNorm weight (q/k norms, the indexer's, the hyper-connection norms, enorm/hnorm) was stored as 1 + w: -1;
  the experts' fused gate_up_proj was split into ffn_gate_exps and ffn_up_exps: joined again;
  the indexer's index_qk_proj was split into indexer.q_proj and indexer.k_proj: joined again;
  fc_embedding and fc_hidden were joined into nextn.eh_proj ([W_e | W_h]): split again;
  the head's hyper_connection_mixer became nextn.hc_head_*; pre_fc_norm_embedding/hidden nextn.enorm/hnorm.
Quantized tensors (Q8_0 in the files seen) are dequantized and stored as BF16.

--ref: the original checkpoint's tensors (an install's mtp folder): each converted tensor is compared with its original.
The original head through the converter at Q8_0 (jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF) measured cosine >= 0.99993
and relative difference <= 0.012 in every tensor, the norms exact; mtp-compare.json says "same" below 0.02.  A
fine-tune of the head stays near that; a wrong mapping does not - exit 4 then (cosine < 0.9 or relative difference >
0.5), so setup falls back to the original's draft layer.  Exit 3: the file has no
complete MTP head.  Nothing here runs a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

NO_HEAD, MISMATCH = 3, 4
MIN_COSINE = 0.9          # a fine-tune of the original head stays far above these; a wrong mapping does not (a norm
MAX_RELATIVE = 0.5        # read without its 1 + w is still cosine ~0.99 but relative difference ~1)
SAME_RELATIVE = 0.02      # Q8_0's rounding: 0.006-0.012 for the original head through PR #27836's converter

# checkpoint name -> (GGUF tensor suffix after blk.N., was stored as 1 + w)
PLAIN = {
    "mtp.hyper_connection_mixer.hc_norm.weight": ("nextn.hc_head_norm.weight", True),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": ("nextn.hc_head_down.weight", False),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": ("nextn.hc_head_up.weight", False),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", False),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", True),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", False),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", False),
    "mtp.layers.0.mlp.experts.down_proj": ("ffn_down_exps.weight", False),
    "mtp.layers.0.mlp.gate.weight": ("ffn_gate_inp.weight", False),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", False),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", False),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", False),
    "mtp.layers.0.mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", False),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", False),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", True),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", False),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", False),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", True),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", True),
    "mtp.layers.0.self_attn.k_norm.weight": ("attn_k_norm.weight", True),
    "mtp.layers.0.self_attn.k_proj.weight": ("attn_k.weight", False),
    "mtp.layers.0.self_attn.o_proj.weight": ("attn_output.weight", False),
    "mtp.layers.0.self_attn.q_norm.weight": ("attn_q_norm.weight", True),
    "mtp.layers.0.self_attn.q_proj.weight": ("attn_q.weight", False),
    "mtp.layers.0.self_attn.v_proj.weight": ("attn_v.weight", False),
    "mtp.pre_fc_norm_embedding.weight": ("nextn.enorm.weight", True),
    "mtp.pre_fc_norm_hidden.weight": ("nextn.hnorm.weight", True),
}
# the checkpoint's shapes (Qwen/Qwen3.8-Flash-Next config: hidden 2560, 4 hyper-connection streams, 512 experts of 640)
SHAPES = {
    "mtp.fc_embedding.weight": [2560, 2560], "mtp.fc_hidden.weight": [2560, 2560],
    "mtp.hyper_connection_mixer.hc_norm.weight": [10240],
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [320, 10240],
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.mlp.experts.down_proj": [512, 2560, 640],
    "mtp.layers.0.mlp.experts.gate_up_proj": [512, 1280, 2560],
    "mtp.layers.0.mlp.gate.weight": [512, 2560],
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": [2560, 640],
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert_gate.weight": [1, 2560],
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": [640, 2560],
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": [128],
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": [128],
    "mtp.layers.0.self_attn.k_norm.weight": [256],
    "mtp.layers.0.self_attn.k_proj.weight": [512, 2560],
    "mtp.layers.0.self_attn.o_proj.weight": [2560, 6144],
    "mtp.layers.0.self_attn.q_norm.weight": [256],
    "mtp.layers.0.self_attn.q_proj.weight": [12288, 2560],
    "mtp.layers.0.self_attn.v_proj.weight": [512, 2560],
    "mtp.pre_fc_norm_embedding.weight": [2560],
    "mtp.pre_fc_norm_hidden.weight": [10240],
}


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits, rounded to nearest even (as torch's .to(torch.bfloat16))."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def as_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class Head:
    """The MTP block's tensors in a GGUF: blk.N.* where N is the block that has nextn.eh_proj."""

    def __init__(self, path: Path):
        self.r = gguf.GGUFReader(str(path))
        eh = [t.name for t in self.r.tensors if t.name.endswith(".nextn.eh_proj.weight")]
        if len(eh) != 1:
            raise LookupError(f"{path.name} has {'no' if not eh else len(eh)} MTP head" + ("" if not eh else "s")
                              + " (blk.N.nextn.eh_proj.weight)")
        self.blk = eh[0].split(".")[1]
        self.t = {t.name[len(f"blk.{self.blk}."):]: t for t in self.r.tensors if t.name.startswith(f"blk.{self.blk}.")}

    def need(self, suffix: str):
        if suffix not in self.t:
            raise LookupError(f"blk.{self.blk}.{suffix} is not in the file")
        return self.t[suffix]

    def f32(self, suffix: str, rows: slice | int | None = None) -> np.ndarray:
        """The tensor as float32 in numpy (checkpoint) order; `rows`: only that slice of its outermost axis."""
        t = self.need(suffix)
        data = t.data if rows is None else t.data[rows]
        return np.asarray(dequantize(np.asarray(data), t.tensor_type), dtype=np.float32)


def convert(src: Path, out: Path) -> list[dict]:
    h = Head(src)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    manifest = []

    def put(name: str, bits: np.ndarray):
        shape = SHAPES[name]
        if bits.size != int(np.prod(shape)):
            raise ValueError(f"{name}: {bits.size} values, the checkpoint has {shape}")
        f = out / "tensors" / f"{name}.bin"
        bits.reshape(-1).tofile(f)
        manifest.append({"name": name, "dtype": "BF16", "shape": shape, "bytes": int(bits.size) * 2,
                         "file": f"tensors/{name}.bin", "sha256": sha256_of(f), "source": src.name})
        print(f"{name}: {shape}", flush=True)

    for name, (suffix, plus_one) in PLAIN.items():
        if name.endswith("experts.down_proj"):
            continue
        x = h.f32(suffix)
        put(name, bf16(x - 1.0 if plus_one else x))
    eh = h.f32("nextn.eh_proj.weight").reshape(2560, 5120)
    put("mtp.fc_embedding.weight", bf16(eh[:, :2560]))
    put("mtp.fc_hidden.weight", bf16(eh[:, 2560:]))
    put("mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        bf16(np.concatenate([h.f32("indexer.q_proj.weight"), h.f32("indexer.k_proj.weight")], axis=0)))
    # the experts, one at a time (gate_up is 6.7 GB as float32)
    gate, up, down = h.need("ffn_gate_exps.weight"), h.need("ffn_up_exps.weight"), h.need("ffn_down_exps.weight")
    n_exp = gate.data.shape[0]
    for name, parts in (("mtp.layers.0.mlp.experts.gate_up_proj", ("ffn_gate_exps.weight", "ffn_up_exps.weight")),
                        ("mtp.layers.0.mlp.experts.down_proj", ("ffn_down_exps.weight",))):
        shape = SHAPES[name]
        if n_exp != shape[0] or any(h.need(p).data.shape[0] != n_exp for p in parts):
            raise ValueError(f"{name}: the file's experts are not {shape}")
        f = out / "tensors" / f"{name}.bin"
        with open(f, "wb") as w:
            for e in range(n_exp):
                x = np.concatenate([h.f32(p, e) for p in parts], axis=0)
                if list(x.shape) != shape[1:]:
                    raise ValueError(f"{name}: expert {e} is {list(x.shape)}, the checkpoint's {shape[1:]}")
                bf16(x).tofile(w)
        manifest.append({"name": name, "dtype": "BF16", "shape": shape, "bytes": int(np.prod(shape)) * 2,
                         "file": f"tensors/{name}.bin", "sha256": sha256_of(f), "source": src.name})
        print(f"{name}: {shape}", flush=True)
    del gate, up, down
    manifest.sort(key=lambda t: t["name"])
    (out / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def compare(out: Path, ref: Path, sample: int = 8) -> tuple[float, float]:
    """Each converted tensor against the original's (the first `sample` experts of the expert tensors): the lowest
    cosine and the largest relative difference."""
    mine = {t["name"]: t for t in json.loads((out / "mtp-manifest.json").read_text())}
    theirs = {t["name"]: t for t in json.loads((ref / "mtp-manifest.json").read_text())}
    low, high = 1.0, 0.0
    for name in sorted(SHAPES):
        a, b = mine[name], theirs.get(name)
        if b is None:
            raise LookupError(f"{ref} has no {name}")
        shape = SHAPES[name]
        n = int(np.prod(shape[1:])) * sample if len(shape) == 3 else None
        x = as_f32(np.fromfile(out / a["file"], dtype="<u2", count=n if n else -1)).astype(np.float64)
        y = as_f32(np.fromfile(ref / b["file"], dtype="<u2", count=n if n else -1)).astype(np.float64)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            x, y = x + 1.0, y + 1.0                    # compared as the scales they are (1 + w)
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
        rel = float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30))
        low, high = min(low, cos), max(high, rel)
        print(f"  {name}: cosine {cos:.5f}, relative difference {rel:.4f}", flush=True)
    return low, high


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="a GGUF with the MTP head (a draft file, or the model part that has it)")
    ap.add_argument("--out", required=True, help="the folder for tensors/ and mtp-manifest.json (as mtp_fetch.py)")
    ap.add_argument("--ref", help="the original checkpoint's tensors (an install's mtp folder), to compare with")
    a = ap.parse_args()
    out = Path(a.out)
    try:
        convert(Path(a.gguf), out)
    except LookupError as e:
        print(f"no MTP head: {e}", file=sys.stderr)
        return NO_HEAD
    if a.ref and (Path(a.ref) / "mtp-manifest.json").exists():
        print(f"compared with {a.ref}:")
        low, high = compare(out, Path(a.ref))
        same = high < SAME_RELATIVE
        (out / "mtp-compare.json").write_text(json.dumps({"ref": str(a.ref), "lowest_cosine": low,
                                                          "largest_relative_difference": high, "same": same}),
                                              encoding="utf-8")
        print(f"lowest cosine {low:.5f}, largest relative difference {high:.4f}"
              + (": the original head (within the file's rounding)" if same else ""))
        if low < MIN_COSINE or high > MAX_RELATIVE:
            print(f"a tensor is far from the original's (cosine {low:.3f}, relative difference {high:.3f}): not a "
                  "Flash-Next MTP head as this tool reads it", file=sys.stderr)
            return MISMATCH
    return 0


if __name__ == "__main__":
    sys.exit(main())
