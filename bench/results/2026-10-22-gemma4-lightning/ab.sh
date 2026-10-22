#!/bin/bash
# ab.sh: the committed momoe-cuda (head) against the working tree's, on the 4070 Ti: tokens the same? speed
S=/tmp/claude-1000/-usr3-local-src-Strata/0b446e8f-0de0-4650-b892-6f030f7bd444/scratchpad
cd /usr3/local/src/momoe; export CUDA_VISIBLE_DEVICES=GPU-ba11890a-f171-b661-62fa-079fd38b10f1
run() {  # name binary model prompt
  bash $S/mr/safe.sh -m 4 -t 900 $2 -m $3 --ctx 8192 --tokens-file $4 -n ${N:-200} > $S/nl/ab-$1.out 2> $S/nl/ab-$1.err
  echo "$1 rc=$? $(grep -oE 'second half of the answer: [0-9.]+ tok/s' $S/nl/ab-$1.err)"
}
while read name model prompt; do
  [ -z "$name" ] && continue
  for r in ${RUNS:-1 2}; do run $name-old $S/head/build-cuda/momoe-cuda $model $prompt; run $name-new build-cuda/momoe-cuda $model $prompt; done
  if cmp -s <(grep tokens: $S/nl/ab-$name-old.out) <(grep tokens: $S/nl/ab-$name-new.out); then echo "$name: same tokens"; else echo "$name: TOKENS DIFFER"; python3 $S/flint/same.py <(grep tokens: $S/nl/ab-$name-old.out | cut -d' ' -f2-) $S/nl/ab-$name-new.out; fi
done <<LIST
bolt models/bolt35-UD-Q4_K_XL/UD-Q4_K_XL/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q4_K_XL-00001-of-00003.gguf $S/zref/bolt.prompt
ornith15 models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf $S/zref/qwen36.prompt
flint models/flint-UD-Q4_K_XL/flint-UD-Q4_K_XL.gguf $S/flint/ref-q4.prompt
LIST
