#!/bin/bash
# zll.sh: zephyr (M4 Air 32 GB), MoMoE against llama.cpp 99db719 (the build the M1 Ultra's numbers used) on the same
# chat prompt (the model's own template, rendered by llama-server), greedy 400 tokens, everything in memory:
#   llama-server: no drafting, then --spec-type draft-mtp 2/3/4 (models with an MTP layer); two requests each
#   momoe-metal: setup's config (its drafting), then without drafting; two runs each
# One program at a time under macsafe; COOL s between runs (a fanless Air).  -> /tmp/zll.txt
M=/usr/local/src/momoe; T=/tmp/mn/tools; L=/tmp/llcpp/build/bin/llama-server; COOL=${COOL:-90}; cd $M
out() { echo "$*" | tee -a /tmp/zll.txt; }
out "#### $(date '+%F %H:%M') $($L --version 2>&1 | head -1)"
for x in "qwen36-ud-iq4_xs 4" "ornith-32gb 4" "lfm25-32gb none"; do
  set -- $x; CFG=$1; DR=$2
  read -r MODEL CTX ARGS NODRAFT < <(python3 - config/$CFG.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
nd, skip = [], 0
for x in a:
    if skip: skip -= 1; continue
    if x in ("--mtp", "--mtp-gguf", "--draft-vocab"): skip = 1; continue
    nd.append(x)
print(a[a.index("-m") + 1], a[a.index("--ctx") + 1], "\x1f".join(a), "\x1f".join(nd))
PY
)
  IFS=$'\x1f' read -r -a A <<< "$ARGS"; IFS=$'\x1f' read -r -a B <<< "$NODRAFT"
  out "=== $CFG ($(basename $MODEL)), context $CTX"
  for D in none 2 3 4; do
    [ $D != none ] && [ $DR = none ] && continue
    X=""; [ $D != none ] && X="--spec-type draft-mtp --spec-draft-n-max $D"
    sleep $COOL
    $T/macsafe.sh -m 2 -t 1200 $L -m $MODEL -ngl 99 -c $CTX --port 8099 --jinja $X > /tmp/zll-srv.log 2>&1 &
    S=$!
    for i in $(seq 300); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; kill -0 $S 2>/dev/null || break; sleep 1; done
    if [ ! -f /tmp/zll-$CFG.prompt ]; then                   # the prompt: the model's template, and its token ids for momoe
      python3 - $CFG <<'PY'
import json, sys, urllib.request
def post(p, b):
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099" + p, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))
msg = [{"role": "user", "content": "Write a Python function that merges two sorted lists into one sorted list, with a docstring, and explain it briefly."}]
text = post("/apply-template", {"messages": msg})["prompt"]
toks = post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
open(f"/tmp/zll-{sys.argv[1]}.txt", "w").write(text)
open(f"/tmp/zll-{sys.argv[1]}.prompt", "w").write(",".join(map(str, toks)))
PY
    fi
    R=$(python3 $T/lsrv_speed.py 8099 "$(cat /tmp/zll-$CFG.txt)" 400 2 2>&1 | tr "\n" " ")
    ACC=$(grep -E "draft acceptance" /tmp/zll-srv.log | tail -1 | grep -oE "draft acceptance = [0-9.]+")
    out "llama.cpp draft=$D: $R $ACC"
    kill $S; wait $S 2>/dev/null
    grep -E "macsafe: .*killing" /tmp/zll-srv.log | head -1
  done
  for v in config nodraft; do
    for r in 1 2; do
      sleep $COOL
      if [ $v = config ]; then C=("${A[@]}"); else C=("${B[@]}"); fi
      $T/macsafe.sh -m 2 -t 1200 build-metal/momoe-metal "${C[@]}" --tokens-file /tmp/zll-$CFG.prompt -n 400 > /tmp/zll-m.out 2> /tmp/zll-m.err
      out "momoe-metal $v run $r (rc=$?): $(grep -E 'second half|MTP up' /tmp/zll-m.err | sed 's/momoe-metal: //' | cut -c1-110 | tr '\n' ' ') $(grep macsafe: /tmp/zll-m.err | tail -1)"
    done
  done
done
out "#### done $(date +%H:%M)"
