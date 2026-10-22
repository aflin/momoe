#!/bin/bash
# tmo.sh MODEL REF [N]: the /tmp/mm-g4 momoe-cuda on mo through run-safe.sh: speed x2, greedy tokens, KL against REF
M=$1; R=$2; N=${3:-299}; export EXE=/tmp/mm-g4/build-cuda/momoe-cuda MODEL=$M
for r in 1 2; do
  bash /usr/local/src/momoe/run-safe.sh g4 --tokens-file $R.prompt -n $N > /tmp/g4s.txt 2>&1
  echo "run $r: $(grep -E 'second half|expert pool [0-9]+ slots = .* of|expert path' /tmp/m-g4.err | tail -3 | sed 's/momoe-[a-z]*: //' | cut -c1-110 | tr '\n' ';') $(grep -E 'REFUSED|KILL' /tmp/g4s.txt)"
done
python3 - $R.tok /tmp/m-g4.out <<'PY'
import sys
r = open(sys.argv[1]).read().split(); l = [x for x in open(sys.argv[2]) if x.startswith("tokens:")]
m = l[0].split()[1:] if l else []; n = min(len(r), len(m))
print("  greedy vs llama.cpp:", next((i for i, (a, b) in enumerate(zip(r, m)) if a != b), n), "of", n)
PY
MOMOE_KL_REF=$R.kl bash /usr/local/src/momoe/run-safe.sh g4k --tokens-file $R.prompt -n $N > /dev/null 2>&1
echo "  $(grep -oE 'KL against.*' /tmp/m-g4k.err)"
