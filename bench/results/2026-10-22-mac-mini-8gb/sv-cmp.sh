#!/bin/bash
# sv-cmp.sh CONFIG [llama-bench placements...]: on savannah, momoe (the config's own args) against llama.cpp a37cc46.
# Speed: momoe 300 tokens x2 (the answer's second half); llama-bench tg64 per placement.  Correctness: a 48-token greedy
# completion of a plain prompt from momoe, llama.cpp on the CPU (the reference: no Metal) and llama.cpp on the GPU.
cd /usr/local/src/momoe
C=config/$1.json; shift
PY=.venv/bin/python
read EXE ARGS <<< "$($PY -c "import json,sys;c=json.load(open('$C'));print(c['exe'],' '.join(c['args']))")"
M=$($PY -c "import json;a=json.load(open('$C'))['args'];print(a[a.index('-m')+1])")
TOK=$($PY -c "import json;print(json.load(open('$C'))['tokenizer'])")
P="The history of the bicycle began in the early nineteenth century, when"
to() { perl -e 'alarm shift; exec @ARGV' "$@"; }
$PY - "$TOK" "$P" > /tmp/cmp.prompt <<'PY'
import json, sys
sys.path.insert(0, "tools"); import strata_tokenizer as ST
from pathlib import Path
t = Path(sys.argv[1]); v = json.loads((t / "vocab.json").read_text()); toks = [None] * len(v)
for k, i in v.items(): toks[i] = k
mj = t / "merges.json"
merges = json.loads(mj.read_text()) if mj.exists() else (t / "merges.txt").read_text().split("\n")
cfg = json.loads((t / "tokenizer.json").read_text())
tk = ST.Tokenizer(toks, merges, json.loads((t / "token_type.json").read_text()), pre=cfg.get("pre") or "qwen35")
ids = tk.encode(sys.argv[2])
bos = cfg.get("bos_token")
if bos: ids = [v[bos]] + ids
print(",".join(map(str, ids)))
PY
for r in 1 2; do
  to 1500 $EXE $ARGS --tokens-file /tmp/cmp.prompt -n 300 > /tmp/cmp.out 2> /tmp/cmp.err
  echo "momoe run $r rc=$?: $(grep -E 'second half' /tmp/cmp.err | sed 's/momoe-[a-z]*: //' | cut -c1-80) $(grep -m1 -iE 'abort|not found|GGML_ASSERT' /tmp/cmp.err | cut -c1-100)"
done
to 900 $EXE $ARGS --tokens-file /tmp/cmp.prompt -n 48 > /tmp/cmp.out 2> /dev/null
$PY - "$TOK" /tmp/cmp.out > /tmp/cmp.momoe.txt <<'PY'
import json, sys
sys.path.insert(0, "tools"); import strata_tokenizer as ST
from pathlib import Path
t = Path(sys.argv[1]); v = json.loads((t / "vocab.json").read_text()); toks = [None] * len(v)
for k, i in v.items(): toks[i] = k
mj = t / "merges.json"
merges = json.loads(mj.read_text()) if mj.exists() else (t / "merges.txt").read_text().split("\n")
cfg = json.loads((t / "tokenizer.json").read_text())
tk = ST.Tokenizer(toks, merges, json.loads((t / "token_type.json").read_text()), pre=cfg.get("pre") or "qwen35")
l = [x for x in open(sys.argv[2]) if x.startswith("tokens:")]
print(tk.decode([int(x) for x in l[0].split()[1:]]) if l else "(no tokens)")
PY
L=/tmp/llcpp/build/bin
for g in "-ngl 0" "${GPU:--ngl 99}"; do
  to 1500 $L/llama-completion -m $M -p "$P" -n 48 --temp 0 $g --no-warmup -no-cnv > /tmp/cmp.ll.txt 2> /tmp/cmp.ll.err < /dev/null
  echo "llama.cpp text ($g) rc=$?" > /dev/null
  cp /tmp/cmp.ll.txt "/tmp/cmp.ll$(echo $g | tr -d ' -').txt"
done
$PY - "$P" /tmp/cmp.momoe.txt /tmp/cmp.llngl0.txt "/tmp/cmp.ll$(echo ${GPU:--ngl 99} | tr -d ' -').txt" <<'PY'
import sys
p = sys.argv[1]
def gen(f):
    s = open(f, errors="replace").read()
    return s[s.find(p) + len(p):] if p in s else s
ref = gen(sys.argv[3])
for name, f in (("momoe", sys.argv[2]), ("llama.cpp GPU", sys.argv[4])):
    g = gen(f); n = next((i for i, (a, b) in enumerate(zip(g, ref)) if a != b), min(len(g), len(ref)))
    print(f"  {name}: same as llama.cpp on the CPU for {n} of {len(ref)} characters")
print("  CPU reference:", repr(ref[:150]))
PY
for a in "$@"; do
  to 1500 $L/llama-bench -m $M $a -p 0 -n 64 -r 1 > /tmp/lb.out 2> /tmp/lb.err
  echo "llama-bench $a: rc=$? $(grep -E 'tg64' /tmp/lb.out | grep -oE '[0-9.]+ ± [0-9.]+' | tail -1) $(grep -m1 -iE 'error|abort|failed|not found|ASSERT' /tmp/lb.err | cut -c1-120)"
done
