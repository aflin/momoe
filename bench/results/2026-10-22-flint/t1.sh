#!/bin/bash
# t1.sh: Flint on the 4070 Ti, both sizes: speed x2, greedy tokens and KL against llama.cpp's reference
S=/tmp/claude-1000/-usr3-local-src-Strata/0b446e8f-0de0-4650-b892-6f030f7bd444/scratchpad
cd /usr3/local/src/momoe
export CUDA_VISIBLE_DEVICES=GPU-ba11890a-f171-b661-62fa-079fd38b10f1
for q in ${QS:-UD-IQ3_XXS UD-Q4_K_XL}; do
  M=models/flint-$q/flint-$q.gguf; R=$S/flint/ref-iq3; [ $q = UD-Q4_K_XL ] && R=$S/flint/ref-q4
  for r in 1 2; do
    env ${ENVS:-X=1} bash $S/mr/safe.sh -m 4 -t 900 build-cuda/momoe-cuda -m $M --ctx ${CTX:-8192} --tokens-file $R.prompt -n 300 > $S/flint/h.out 2> $S/flint/h.err
    echo "$q: $(grep -E 'second half|per token: graph|expert path' $S/flint/h.err | sed 's/momoe-[a-z]*: //' | cut -c1-120 | tr '\n' ';') $(grep -iE ' error|abort|failed' $S/flint/h.err | head -1)"
  done
  python3 $S/flint/same.py $R.tok $S/flint/h.out
  env ${ENVS:-X=1} MOMOE_KL_REF=$R.kl bash $S/mr/safe.sh -m 4 -t 900 build-cuda/momoe-cuda -m $M --ctx ${CTX:-8192} --tokens-file $R.prompt -n 300 > /dev/null 2> $S/flint/hk.err
  echo "  $(grep -oE 'KL against.*' $S/flint/hk.err)"
done
