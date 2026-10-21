#!/bin/bash
# llref2.sh MODEL N OUT [llama-server args...] : on mo, llama.cpp greedy reference for one chat prompt (the model's
# own chat template).  Writes OUT.prompt (the prompt's token ids, for momoe --tokens-file), OUT.tok (llama's N greedy
# tokens) and OUT.kl (a MOMOE_KL_REF file: llama's top-20 log-probabilities each step, the rest -60000).
# Refuses beside a running engine; kills llama-server when MemAvailable falls under 8 GB.
M=$1; N=$2; OUT=$3; shift 3; [ $# -eq 0 ] && set -- -ngl 999
if pgrep -x momoe-cuda > /dev/null || pgrep -x llama-server > /dev/null; then echo "REFUSED: an engine is running"; exit 2; fi
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
for i in $(seq 120); do [ $(avail) -ge 100 ] && break; sleep 1; done     # a run before it: the GPU's memory comes back slowly
if [ $(avail) -lt 100 ]; then echo "REFUSED: only $(avail) GB available"; exit 2; fi
/usr/local/src/momoe/llama.cpp/build/bin/llama-server -m "$M" -c 8192 --port 8091 --host 127.0.0.1 --jinja "$@" > $OUT.log 2>&1 &
pid=$!
( low=999; while kill -0 $pid 2>/dev/null; do a=$(avail); [ $a -lt $low ] && low=$a
    if [ $a -lt 8 ]; then echo "WATCHDOG: MemAvailable $a GB, llama-server killed" >> $OUT.log; kill -9 $pid; fi
    echo $low > $OUT.low; sleep 0.5; done ) &
trap "kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null" EXIT
for i in $(seq 900); do curl -s 127.0.0.1:8091/health | grep -q ok && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
/usr/local/src/momoe/.venv/bin/python - "$M" "$N" "$OUT" <<'PY'
import json, sys, struct, urllib.request
import numpy as np
model, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8091" + path, json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=7200))
msg = [{"role": "user", "content": "Write a Python function that merges two sorted lists into one sorted list, "
                                   "with a docstring, and explain it briefly."}]
text = post("/apply-template", {"messages": msg})["prompt"]
toks = post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
open(out + ".prompt", "w").write(",".join(map(str, toks)))
r = post("/completion", {"prompt": toks, "n_predict": n, "temperature": 0, "top_k": 1, "return_tokens": True,
                         "cache_prompt": False, "n_probs": 20})
cp = r["completion_probabilities"]
sys.path.insert(0, "/usr/local/src/momoe/tools")
from gguf_reader import GGUFFile
V = len(GGUFFile(model).metadata["tokenizer.ggml.tokens"])
open(out + ".tok", "w").write(" ".join(map(str, r["tokens"])))
with open(out + ".kl", "wb") as f:
    for i in range(1, len(cp)):          # momoe's step i-1 is fed token i-1 and predicts token i: llama's step i
        lp = np.full(V, -60000, dtype=np.float16)
        for t in cp[i]["top_logprobs"]:
            lp[t["id"]] = t["logprob"]
        f.write(struct.pack("<i", r["tokens"][i - 1])); f.write(lp.tobytes())
t = r.get("timings", {})
print(f"{len(toks)} prompt tokens, {len(cp)} generated ({t.get('predicted_per_second', 0):.1f} tok/s), V {V}: "
      f"{r['content'][:80]!r}")
PY
echo "lowest MemAvailable $(cat $OUT.low) GB"
