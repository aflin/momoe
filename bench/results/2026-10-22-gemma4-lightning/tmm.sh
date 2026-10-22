#!/bin/bash
# tmm.sh MODEL REF [N]: momoe-metal (this copy): speed x2 (the answer's second half), greedy tokens and KL against
# llama.cpp's REF (.prompt .tok .kl)
cd /tmp/mm-g4; M=$1; R=$2; N=${3:-299}
for r in 1 2; do
  env ${ENVS:-X=1} build-metal/momoe-metal -m $M --ctx ${CTX:-8192} --tokens-file $R.prompt -n $N > /tmp/g4m.out 2> /tmp/g4m.err
  echo "run $r rc=$?: $(grep -E 'second half|expert pool [0-9]+ slots = .* of|preload' /tmp/g4m.err | tail -3 | sed 's/momoe-[a-z]*: //' | cut -c1-120 | tr '\n' ';') $(grep -iE ' error|abort|failed' /tmp/g4m.err | head -1)"
done
python3 - $R.tok /tmp/g4m.out <<'PY'
import sys
r = open(sys.argv[1]).read().split(); l = [x for x in open(sys.argv[2]) if x.startswith("tokens:")]
m = l[0].split()[1:] if l else []; n = min(len(r), len(m))
print("  greedy vs llama.cpp:", next((i for i, (a, b) in enumerate(zip(r, m)) if a != b), n), "of", n)
PY
env ${ENVS:-X=1} MOMOE_KL_REF=$R.kl build-metal/momoe-metal -m $M --ctx ${CTX:-8192} --tokens-file $R.prompt -n $N > /dev/null 2> /tmp/g4mk.err
echo "  $(grep -oE 'KL against.*' /tmp/g4mk.err) $(grep -iE ' error|abort|shorter' /tmp/g4mk.err | head -1)"
