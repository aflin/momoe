# Expert statistics in the logs

A mixture-of-experts model picks a few experts for every token in every layer. How fast MoMoE runs a model depends
mostly on where those experts are when they are needed: in the GPU's memory, or somewhere slower (RAM, the SSD, the
CPU). The engines write that to their log (`logs/<model>.log`), per request and while a request runs, in one format.

## The lines

After each request's `prompt ... tokens` line:

```
momoe serve: experts: prompt 68.6% in the pool (11016 of 16063; 15583 of them in prompt batches); 41930 MiB read in all (misses, read-ahead, pool refills), 4.7 s waited for reads; writing 96.5% in the pool (13205 of 13680); 16650 MiB read in all (misses, read-ahead, pool refills), 1.0 s waited for reads
```

Every 60 seconds while the engine works, for the time since the last such line (a long prompt or a long answer shows
it while it runs; nothing is written while the engine is idle):

```
momoe serve: experts, the last 60 s: 96.7% in the pool (66329 of 68583; 2751 of them in prompt batches); ...
```

`MOMOE_HIT_LOG_S` sets the interval in seconds (in the config's `"env"`, for example `"MOMOE_HIT_LOG_S": "30"`); `0`
turns the periodic line off. The line after each request stays.

**prompt** is reading the prompt, **writing** is generating the answer (with its draft tokens). They are counted apart
because they behave differently: a prompt needs most of a layer's experts at once, an answer a few at a time.

## What is counted

One count is one expert that one layer needed for one batch of tokens: while writing, a batch is one step (a token
and its drafts); while reading a prompt, a batch is one prompt chunk. An expert that several tokens of the same batch
use is counted once, so the totals depend on the chunk size (`--chunk` in the config: 512 tokens on a Mac; on Linux
1024 for most models, 4096 for some): compare percentages, not totals, and compare the same machine and settings.

| Engine | Used for | "Fast" means | The rest |
| --- | --- | --- | --- |
| `momoe-cuda`, `momoe-metal` (MoMoE's ggml engine) | every model on a Mac, the DGX Spark and Jetson; Nemotron, GLM, Ornith, Ling, Flint, Gemma, LFM on a PC | **in the pool**: the experts held in GPU memory | read from the model's file (the SSD, or the OS file cache) |
| `momoe` (Strata's engine) | Qwen3.8-Flash-Next and the Qwen3.6 family on a PC | **on the card**: the GPU's expert cache | copied in from RAM, computed on the CPU, or read by the GPU over PCIe |
| `strata-dsv41` (Strata's DeepSeek engine) | DeepSeek-V4.1-Flash on a PC | - | no expert line: its experts have fixed places (the GPU or the CPU) |

Strata's engine also keeps its own line from upstream, `decode expert cache hit rate: ... (hits / lookups)`: the same
writing figure, but without the experts read over PCIe in the total.

### MoMoE's ggml engine

- **in prompt batches**: how many of the counted experts a big prompt batch needed. Such a batch reads each layer's
  experts into a staging copy: copied from the pool when it holds them (counted as in the pool), else read from the
  file.
- **MiB read in all**: everything read from the model's file in that time, not only the experts that were missing:
  also the experts the engine reads ahead because it predicts the next layer will need them, and the ones it copies
  into the pool because they were used often. A high hit rate with many MiB read is normal: the pool is being kept
  fresh for later tokens.
- **waited for reads**: the time the engine actually stopped and waited for the SSD. This, not the MiB, is what the
  reads cost in speed.

### Strata's engine

- **prompt**: per prompt chunk and layer, the expert groups served from the VRAM cache, copied in from RAM, or computed
  by the CPU (`STRATA_PREFILL_CPU_SHARE`).
- **writing**: the experts found in the GPU's cache; "not on it" are the others, which the CPU computes or the GPU
  reads from RAM over PCIe (`--pcie-frac`; the count says how many went that way). Nothing here is read from the SSD
  unless the RAM does not hold the model.

## Measured examples

DeepSeek-V4.1-Flash IST-DAS 2.00 bpw, 2026-10-29, a fresh 4,247-token prompt, then a 600-token answer from a short one:

| Machine | Pool | Prompt | Writing |
| --- | ---: | ---: | ---: |
| DGX Spark (`momoe-cuda`) | 58 GiB, 47% of the experts | 68.6% in the pool | 98.4% |
| M1 Ultra 64 GB (`momoe-metal`) | 28 GiB, 23% | 85.5% | 93.0% (a 400-token answer) |

The two prompt figures are not comparable with each other: the Spark reads 1,024-token chunks and the Mac 512 (see
above), and the Spark's pool fills differently on unified memory. Each is comparable with itself across runs.
