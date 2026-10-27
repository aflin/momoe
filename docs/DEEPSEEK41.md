# DeepSeek-V4.1-Flash in MoMoE

DeepSeek-V4.1-Flash is a large mixture of experts: 40 layers of 384 experts (6 used for each token), a compressed
sparse attention (a 128-token window plus the best 512 of the compressed rows a learned indexer picks), four-stream
hyper-connections, and an n-gram table (Engram) that two layers read. MoMoE runs the official 2-bit sets from IST-DAS
(ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), pinned in `setup.py` with their sizes and SHA-256.

| Set | Model files | + n-gram table | Disk |
| --- | ---: | ---: | ---: |
| IST-DAS 2.00 bpw (IQ2_XXS), the default | 138 GB | 111 GB | 249 GB (257 GB on a PC, with the 8 GB drafter) |
| IST-DAS 2.25 bpw (IQ2_XS) | 155 GB | 111 GB | 266 GB (274 GB) |
| IST-DAS 1.75 bpw (IQ1_M) | 121 GB | 111 GB | 232 GB (240 GB) |

The n-gram table is one file shared by the three sets (`models/deepseek-v4.1-flash-shared`). Quality against the
original model, upstream's measurement (4,995 positions): perplexity +15.5% at 2.00 bpw (code +8.6%, agent work +5.3%),
+7.3% at 2.25, +21% at 1.75 (weaker on plain prose).

## Two engines

- **A PC with an NVIDIA card:** upstream Strata's DeepSeek engine, `strata-dsv41` (`src/dsv41`, ported by upstream
  from Project Maya's maya-ds4, MIT). Setup compiles it here (`cmake -DSTRATA_BUILD_DSV41=ON`, into `engine-dsv41/`):
  MoMoE has no ready-made engines.
- **An Apple Silicon Mac, and a GPU that shares the system's memory (the DGX Spark):** MoMoE's own `momoe-metal` /
  `momoe-cuda` (`src/metal`, after llama.cpp's DeepSeek-V4.1 graph). It reads the IST-DAS files as they are; the
  experts that do not fit in its pool are read from the SSD, the n-gram rows on the CPU.

```
./setup.sh --family deepseek41 --model IST-DAS-2.00bpw
```

## Measured

| Machine | Engine, config | Chat, writing | 4K prompt | 16K prompt |
| --- | --- | ---: | ---: | ---: |
| PC: RTX 4070 Ti 12 GB, 63 GB RAM, files on a SATA SSD RAID1 | `strata-dsv41`, 8K context | 3.8 tok/s | 81.7 tok/s | - |
| M1 Ultra, 64 GB | `momoe-metal`, setup's config (28 GiB pool, 32K context) | 7.4 tok/s | 82.6 tok/s | 71.9 tok/s |
| DGX Spark, 128 GB shared | `momoe-cuda`, setup's config (a 77 GiB pool, 32K context, DSpark drafting) | 19.2 tok/s | 268 tok/s | 242 tok/s |
| DGX Spark, without DSpark | `momoe-cuda`, an 85 GiB pool | 16.0 tok/s | 255 tok/s | 238 tok/s |

The server's benchmark (`tools/strata_bench.py`), 2 or 3 runs each; the reports are in `bench/results/2026-10-2*-dsv41-*`.
An agent session (turns of 2,800-3,000 new tokens) wrote 5.9 tok/s on the M1 Ultra, its turns starting after about
90 s, and on the Spark 19.6 tok/s with DSpark (14.4 without), after about 22 s. The first token of a short chat: 0.8 s on the Spark, 3.3 s on the
M1 Ultra, 13.4 s on the PC. The prompt speeds on the Spark varied between the two runs (4K: 144 and 366 tok/s).

**Checked:** MoMoE's engine against `strata-dsv41` (the log-probabilities of 64 answer tokens, both fed the same
tokens; greedy, 6 experts, no drafting):

| Prompt | `momoe-cuda` (PC / Spark) KL mean, same top token | `momoe-metal` (M1 Ultra) |
| --- | --- | --- |
| 14-22 tokens (two) | 0.001-0.012, 97-100% | 0.001-0.009, 100% |
| 2,110 tokens | 0.006, 98% | 0.007, 97% |
| 8,000 tokens | 0.017, 95% (Spark) | - |
| 19,999 tokens | 0.016, 95% (Spark) | - |

The difference grows from short prompts to 8K and then stays: upstream's engine keeps the attention's caches in FP8
and FP4, as DeepSeek's own inference code does, and MoMoE's (as llama.cpp's) in F16, so a few of the 512 rows the
sparse attention picks differ on long prompts. The code prompt's greedy answer was the same 42 tokens. The server's
chat, thinking on and off, tool calls (OpenAI and Anthropic) and streaming on both engines; a conversation continued
from the cache after another one ran (1,352 of 1,406 tokens reused) answered exactly as the same prompt read fresh.

## DSpark drafting

DSpark is DeepSeek-V4.1's draft model (Unsloth's `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, 8 GB): three small
blocks that read the model's inner state at blocks 37-39 and propose the next 5 tokens at once; the model checks them
in one pass and keeps the ones it would have written itself, so the answer is the model's. MoMoE's engine runs it
after llama.cpp's graph (`--dspark FILE`; a draft is cut where its confidence head falls under 0.65,
`--dspark-conf`). Setup turns it on on a GPU that shares the system's memory (the Spark) and leaves it off on a Mac,
where its 7.5 GiB would come out of an already small pool (not measured there).

On the Spark: chat 16.0 -> 19.2 tok/s and agent work 14.4 -> 19.6 (the server's benchmark); 1.9-4 tokens a check
(code 4, thinking text 2), 85-90% of the drafts kept. Greedy answers were the same with and without it except at
near ties (a long answer's choice between two phrasings, a word's ending), and matched `strata-dsv41`'s on the
code and thinking prompts; a run with the drafter loaded but no draft kept was identical to one without it. A
failed draft is not paid for with the whole state copied and the accepted tokens run again: only the compressor's
unfinished group is put back (the first version did that and was slower than no drafting on two of three questions).

## Memory

The pool is smaller than for other models of this size: besides the 5.7 GiB of other weights, the engine keeps a
staging copy of one layer's experts (3.7 GiB) for long prompts. On the 64 GB M1 Ultra a 36 GiB pool (46 GiB of GPU
memory in all) froze the machine; setup leaves 7 GiB more free and gives 28 GiB there. A 32 GiB pool wrote 8.9 tok/s
against 8.7 at 28: little to gain from the risk. Run by hand without `--pool-gib`, the engine leaves 15% of the GPU's
working set free when it has to make the pool smaller (a pool that fits beside a 3% margin is left as it is).

On the DGX Spark setup keeps 20 GiB from the pool for the rest of the system (`MOMOE_UMA_RESERVE_GIB` in the config):
with the engine's default 10 the memory available fell to 8 GB while it wrote; with 20 it stayed at 16-18 GB, at the
same speed.

## Not in this version

- DSpark on a Mac (built, not measured: see above).
- Pictures.
- AMD cards (upstream's engine supports the Radeon AI PRO R9700; MoMoE's setup offers NVIDIA only for now).
- More than one card with upstream's engine (not tried here).
