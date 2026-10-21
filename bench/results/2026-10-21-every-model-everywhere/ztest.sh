#!/bin/bash
# ztest.sh CFG REF [N] : zephyr, one model installed by momoe setup (config/CFG.json) against llama.cpp's reference
# /tmp/zref/REF.{prompt,tok,kl}: (1) speed with the config's own args, N tokens (2) KL over N tokens fed llama's, no
# drafting (3) the first 64 greedy tokens (4) the server: chat, tool call (whole, streamed), tool result.
# Results appended to /tmp/zres.txt.
CFG=$1; REF=/tmp/zref/$2; N=${3:-300}; COOL=${COOL:-60}
cd /usr/local/src/momoe
read -r EXE ARGS NODRAFT < <(python3 - config/$CFG.json <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]
nd, skip = [], 0
for i, x in enumerate(a):
    if skip: skip -= 1; continue
    if x in ("--mtp", "--mtp-gguf", "--draft-vocab"): skip = 1; continue
    nd.append(x)
print(c["exe"], "\x1f".join(a), "\x1f".join(nd), sep=" ")
PY
)
IFS=$'\x1f' read -r -a A <<< "$ARGS"; IFS=$'\x1f' read -r -a B <<< "$NODRAFT"
L=/tmp/zt-$CFG; out() { echo "$*" | tee -a /tmp/zres.txt; }
out "=== $CFG $(date '+%F %H:%M') args: ${A[*]:2}"
sleep $COOL
/usr/bin/time -l "$EXE" "${A[@]}" --tokens-file $REF.prompt -n $N > $L.speed.out 2> $L.speed.err
out "speed rc=$? $(grep -E 'second half|MTP up|^momoe-metal: generated' $L.speed.err | sed 's/momoe-metal: //' | tr '\n' ';') $(grep -E 'maximum resident' $L.speed.err | awk '{printf "max RSS %.1f GB", $1/1e9}')"
python3 - $L.speed.out $REF.tok <<'PY' | tee -a /tmp/zres.txt
import sys
m = [l for l in open(sys.argv[1]) if l.startswith("tokens:")]
r = open(sys.argv[2]).read().split()
m = m[0].split()[1:] if m else []
same = next((i for i, (x, y) in enumerate(zip(r, m)) if x != y), min(len(r), len(m), 64))
print(f"greedy vs llama.cpp: {min(same, 64)}/64 the same" + (f" (parts at {same})" if same < 64 else ""))
PY
sleep $COOL
MOMOE_KL_REF=$REF.kl "$EXE" "${B[@]}" --tokens-file $REF.prompt -n $N > $L.kl.out 2> $L.kl.err
out "kl rc=$? $(grep -oE 'KL against.*' $L.kl.err)"
sleep $COOL
./momoe start $CFG > $L.start.txt 2>&1 && python3 /tmp/srvcheck.py 8080 2>&1 | tee -a /tmp/zres.txt
./momoe stop > /dev/null 2>&1
out ""
