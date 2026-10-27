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

The server's benchmark (`tools/strata_bench.py`), 2 or 3 runs each; the reports are in `bench/results/2026-10-26-dsv41-*`.
On the M1 Ultra an agent session wrote 5.9 tok/s, and its turns (2,800-3,000 new tokens each) took about 90 s to
start. The first token of a short chat: 3.3 s on the M1 Ultra, 13.4 s on the PC.

**Checked:** MoMoE's engine against `strata-dsv41` (the log-probabilities of every answer token, both fed the same
tokens; greedy, 6 experts, no drafting), on prompts of 14, 22 and 2,110 tokens: KL mean 0.001-0.010, the same top
token at 97-100% of the steps; the code prompt's greedy answer was the same 42 tokens. The server's chat, thinking on
and off, tool calls (OpenAI and Anthropic) and streaming on both engines.

## Memory on a Mac

The pool is smaller than for other models of this size: besides the 5.7 GiB of other weights, the engine keeps a
staging copy of one layer's experts (3.7 GiB) for long prompts. On the 64 GB M1 Ultra a 36 GiB pool (46 GiB of GPU
memory in all) froze the machine; setup leaves 7 GiB more free and gives 28 GiB there. A 32 GiB pool wrote 8.9 tok/s
against 8.7 at 28: little to gain from the risk. Run by hand without `--pool-gib`, the engine leaves 15% of the GPU's
working set free.

## Not in this version

- Drafting with the DSpark model on MoMoE's engine (upstream's PC engine has it: `--dspark auto`).
- Pictures.
- AMD cards (upstream's engine supports the Radeon AI PRO R9700; MoMoE's setup offers NVIDIA only for now).
- More than one card with upstream's engine (not tried here).
