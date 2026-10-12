# A second, slower GPU for Flash-Next: the layer split, measured

**Status:** Rejected for this PC (2026-10-12). Measured, not built: the layer split is Strata's (docs/MULTI_GPU.md).
Worth measuring again on a pair of cards closer in speed, and for Qwen3.6 / Ornith once they can split.

## The question

This PC has an RTX 4070 Ti (12 GB, the model's card) and an RTX 3060 (12 GB, driving the desktop, which takes about
1.1 GB of it). Does giving Flash-Next part of the 3060 through the layer split make it faster than the 4070 Ti alone?

The split puts each card in charge of a range of layers; each card caches the experts of its own layers only. The
3060 can be limited with `--vram-reserve-later-mib N` (VRAM left unused on every card after the first) and with
`--layer-split K` (the first layer of the later card). Each card also holds its own copy of its layers' dense weights
and its own prompt buffers, so a small share of a card leaves little for experts.

## Method

- Qwen3.8-Flash-Next IQ3_XXS (`packs/iq3_xxs`), Momoe engine 0.1.43 at commit 6d5127ff (built for sm_86 and sm_89),
  `--max-context 65536 --kv int8 --spec 4 --mtp`, `--expert-cache auto`.
- Ryzen 9 5950X, 63 GB RAM, Linux 6.1, driver 580.178.04, CUDA 13.0. The 4070 Ti's PCIe link measured 13.4 GB/s.
- `tools/strata_bench.py --long`: three short chats (100 tokens out), fresh 4K, 16K and 32K prompts, and a 5-turn
  agent session, three runs each, medians. Thinking off, temperature 0. One server start per configuration.
- `"gpu": [1, 0]`: the 4070 Ti first (layers from 0), the 3060 second (the last layers, the head and the draft
  layer).

## Results

| | writing (chat) tok/s | writing (agent) tok/s | 4K prompt tok/s | 16K prompt tok/s | 32K prompt tok/s | decode cache hits | 3060 runs |
|---|---|---|---|---|---|---|---|
| 4070 Ti alone | 60.7 (56.9-70.6) | 69.6 | 1,182 | 2,145 | 2,170 | 68.8% | - |
| + 3060, 4 GB held back, split auto | 62.9 (54.1-67.8) | 66.9 | 591 | 578 | 576 | - | layers 45-47 |
| + 3060, all free VRAM, split auto | 60.6 (53.4-69.5) | 65.3 | failed | 2 of 3 failed | 1,494 | 81.2% | layers 45-47 |
| + 3060, all free VRAM, `--layer-split 42` | 62.7 (56.2-73.2) | 68.9 | 1,101 | 1,997 | 2,267 | 85.3% | layers 42-47 |
| + 3060, all free VRAM, `--layer-split 38` | 63.1 (54.5-72.8) | 67.6 | 1,035 | 1,951 | 2,339 | 89.0% | layers 38-47 |

Cache hits: the mean of the last 15 `decode expert cache hit rate` lines of each engine log.

The writing-speed ranges overlap in every row: no configuration writes measurably faster than the 4070 Ti alone.

## What the logs say

- **The 3060 is about 3x slower per layer.** The split's placement search timed it at 1.38 ms per layer (28 SMs at
  1.88 GHz) against 0.46 ms on the 4070 Ti (60 SMs at 2.62 GHz), so the automatic split gives it the last 3 of 48
  layers.
- **With 4 GB held back** the 3060 had 1.50 GiB for experts (827 slots) against 4.76 GiB (2,910) on the 4070 Ti
  alone. Its small cache cut the prompt chunk of both cards (the engine says so: #1616), which is the likely cause of
  prompts reading at a quarter to half the speed.
- **With all its free VRAM and the automatic split** the 3060 still ran only layers 45-47. Those layers have 1,536
  experts and all of them fit in 2.80 GiB, so about 2.4 GB of the card stayed unused: the layer count, not the VRAM,
  was the limit. The engine logged a near tie with `--layer-split 42` (predicted 2.6% slower; its prediction prices
  decode, not prompts).
- **Splitting by hand uses the card:** at 38 the 3060 runs 10 layers and caches 4.65 GiB of experts (2,587 slots),
  the 4070 Ti 6.44 GiB (4,095): the hit rate goes from 68.8% to 89.0%. Writing speed does not move. The likely reading
  is that the extra hits are paid for by the 3060's slower layers; the experts read from RAM were not the bottleneck
  they are on a card pair of similar speed.

## A bug found on the way

The automatic split with no reserve on the 3060 ran out of VRAM on the first long prompt: `lending the prompt path its
slots failed: prefill: device buffers for a chunk of 4096 tokens do not fit (305 of 11874 MiB free at the failure)`,
then `the engine stopped unexpectedly (exit code 1)`. The automatic placement does not leave room for the prompt
buffers on the later card when nothing is reserved there. Setting `--layer-split` by hand (42, 38) did not hit it.
This is Strata's split code, not a Momoe change; not reported upstream yet.

## Not tested

- Splits earlier than layer 38: they put more layers on the slower card, and the trend from 45 to 38 is flat.
- The 3060 as a helper expert cache instead of a split stage (`--expert-cache-device1 N`, docs/SECOND_GPU.md), which
  the engine suggests for a card this small. The docs report it slower on their machine with an RTX 2080 Ti.
- Other Flash-Next sizes, longer contexts, and repeated server starts (one start per configuration).
- Qwen3.6 / Ornith: they run on one GPU only (no layer split for qwen35moe yet). Their experts are fewer and smaller,
  so a second card could hold most of them; whether that pays off depends on the same per-layer cost.

## Files

Configs `fn-iq3-1gpu.json`, `fn-iq3-2gpu.json`, `fn-iq3-2gpu-full.json`, `fn-iq3-2gpu-k42.json`,
`fn-iq3-2gpu-k38.json` in the Momoe folder (git-ignored), their logs beside them, and the benchmark reports in
`/usr3/local/src/momoe-notes/fnbench/` (outside the repository).
