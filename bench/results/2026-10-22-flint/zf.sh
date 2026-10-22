#!/bin/bash
# zephyr (M4 Air 32 GB), Flint: momoe-metal (/tmp/mm-fix build) KL against llama.cpp's reference (made on firefly),
# greedy tokens, speed x2; llama-bench (llama.cpp a37cc46) beside it.  One at a time under macsafe, 90 s apart.
cd /usr/local/src/momoe; E=/tmp/mm-fix/build-metal/momoe-metal; T=/tmp/mn/tools; B=/tmp/llflint/build/bin/llama-bench
while ! grep -q "#### done" /tmp/dl-flint.log; do sleep 20; done
cat /tmp/dl-flint.log
for x in "UD-IQ3_XXS iq3 " "UD-Q4_K_XL q4 --pool-gib 16 --nocache"; do
  set -- $x; q=$1; r=$2; shift 2; M=models/flint-$q/flint-$q.gguf; R=/tmp/fref/ref-$r
  echo "== momoe-metal $q $*"
  sleep 90; MOMOE_KL_REF=$R.kl $T/macsafe.sh -m 2 -t 1500 $E -m $M --ctx 8192 "$@" --tokens-file $R.prompt -n 300 > /tmp/zf.out 2> /tmp/zf.err
  echo "kl rc=$?: $(grep -oE 'KL against.*' /tmp/zf.err)"
  for n in 1 2; do
    sleep 90; $T/macsafe.sh -m 2 -t 1500 $E -m $M --ctx 8192 "$@" --tokens-file $R.prompt -n 300 > /tmp/zf.out 2> /tmp/zf.err
    echo "run $n rc=$?: $(grep -E 'second half|expert pool [0-9]+ slots = .* of|generated' /tmp/zf.err | tail -3 | sed 's/momoe-metal: //' | cut -c1-140 | tr '\n' ';') $(grep macsafe: /tmp/zf.err | tail -1)"
  done
  python3 -c "
r=open('$R.tok').read().split(); m=[l for l in open('/tmp/zf.out') if l.startswith('tokens:')][0].split()[1:]
print('greedy vs llama.cpp:', next((i for i,(a,b) in enumerate(zip(r,m)) if a!=b), min(len(r),len(m))), 'of', min(len(r),len(m)))"
  NG=99; [ $q = UD-Q4_K_XL ] && NG=0                       # (25 GB: over Metal's 26.8 GB working set with its buffers)
  sleep 90; echo "llama-bench -ngl $NG: $($T/macsafe.sh -m 2 -t 1500 $B -m $M -ngl $NG -fa 1 -p 512 -n 128 -r 2 2>&1 | grep -E 'pp512|tg128|macsafe: .*killing' | sed 's/  */ /g' | cut -c1-150 | tr '\n' ';')"
done
echo "#### done"
