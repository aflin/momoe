#!/bin/bash
# tm.sh MODEL REF [N]: momoe-cuda on the 4070 Ti: speed x2 (2nd half), greedy tokens and KL against llama.cpp's REF
S=/tmp/claude-1000/-usr3-local-src-Strata/0b446e8f-0de0-4650-b892-6f030f7bd444/scratchpad
cd /usr3/local/src/momoe; export CUDA_VISIBLE_DEVICES=GPU-ba11890a-f171-b661-62fa-079fd38b10f1
M=$1; R=$2; N=${3:-300}
for r in 1 2; do
  env ${ENVS:-X=1} bash $S/mr/safe.sh -m 4 -t 900 build-cuda/momoe-cuda -m $M --ctx ${CTX:-32768} --chunk 1024 --tokens-file $R.prompt -n $N > $S/nl/h.out 2> $S/nl/h.err
  echo "run $r rc=$?: $(grep -E 'second half|expert pool [0-9]+ slots = .* of|expert path|hybrid:' $S/nl/h.err | tail -4 | sed 's/momoe-[a-z]*: //' | cut -c1-130 | tr '\n' ';') $(grep -iE ' error|abort|failed|die' $S/nl/h.err | head -1)"
done
python3 $S/flint/same.py $R.tok $S/nl/h.out
env ${ENVS:-X=1} MOMOE_KL_REF=$R.kl bash $S/mr/safe.sh -m 4 -t 900 build-cuda/momoe-cuda -m $M --ctx ${CTX:-32768} --chunk 1024 --tokens-file $R.prompt -n $N > /dev/null 2> $S/nl/hk.err
echo "  $(grep -oE 'KL against.*' $S/nl/hk.err)"
