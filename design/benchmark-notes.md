# Benchmark notes

The details behind the README's speed tables: how each column was measured, what drafting does per model, and what
the engines' speed-ups were measured at. The raw numbers are in [bench/results](../bench/results).

## How the tables were measured

2026-10-19 unless named; the M1 Ultra and Spark columns and LFM2.5-8B on the PC 2026-10-20; Ornith-122B and the M4
Air column 2026-10-21 ([its results](../bench/results/2026-10-21-every-model-everywhere/README.md)); Flint, Gemma 4,
Nemotron 3.5 Lightning and the 8 GB Mac mini 2026-10-22. `momoe-cuda` / `momoe-metal` built with ggml from llama.cpp
b11724.

- **Strata's engine on the PC:** `tools/strata_bench.py`, three runs, medians: "chat decode" in the table.
- **`momoe-cuda` and `momoe-metal`:** the engine on its own, greedy, the second half of a 300-400-token answer, with
  the arguments setup wrote (drafting where they have it); "server" rows: the server's own speed over a long chat.
- **The Spark:** one model at a time under a watchdog. GLM-5.3-Flash with `--mtp 3`, 160 tokens; its server and the
  runs without drafting, 2026-10-17. UD-IQ3_XXS (113 GiB, setup's choice on the Spark) is a quarter smaller than
  UD-IQ4_XS (146 GiB): nearly all its experts fit in memory (97.6% of expert reads from the pool) and it writes 1.3-1.4
  times as fast, more with drafting.
- **The Jetson:** 16K context. llama.cpp b11724, `llama-bench` tg128.
- **The Macs:** setup's configs: every expert in memory where it fits beside the KV cache (Flash-Next IQ3_XXS and Swift
  at 32K context, the others at 64K), else an expert pool and the rest from the SSD (Unsloth's Flash-Next: 36 GiB on the
  M1 Ultra; GLM: 29 GiB; on the M4 Air, Flash-Next IQ3_XXS 11 GiB and Ornith 14 GiB, from the memory free when setup
  ran, with Firefox open). The Flash-Next family drafts with Strata's draft layer (its ~5 GB of MTP tensors from Qwen's
  checkpoint, packed Q2_0, as Strata's setup does on a PC) unless every expert fits only without it (Flash-Next IQ3_XXS
  on the M1 Ultra, before graphs were kept: 29.7 tok/s without it, 24.3 drafting from a 37 GiB pool; Unsloth's
  UD-IQ4_XS: 19.0 drafting from 36 GiB, 12.4 without from 39). Unsloth's Flash-Next and GLM on the M1 Ultra: 2026-10-19
  (an expert pool: no graphs kept). The files larger than Metal's working set (48 GiB on the M1 Ultra) were not run in
  llama.cpp: that took a MacBook down.
- **The M4 Air as an 8 GB Mac:** setup's 8 GB config (a 3 GiB pool, 16K context; setup run with
  `MOMOE_MAC_MEMORY_GIB=8`), with 10 GiB locked away (`mlock`) and about 5 GB left free beside macOS and the open
  programs; with all 32 GB the same runs wrote 43.3 and 12.0 tok/s (14.8 for the 24B before; the same with the
  wait-for-event thread off). It does not simulate an 8 GB Mac's smaller Metal limit; the real 8 GB Mac mini is in
  [its results](../bench/results/2026-10-22-mac-mini-8gb/README.md).
- **Before 2026-10-19:** Ornith on the M4 Air, its 16 GB simulation and llama.cpp beside them are in
  [mac-port.md](mac-port.md).

Correctness: every model's greedy output was compared with llama.cpp's (48 or 64 tokens: the same, or parting only
where the top two tokens are within a few hundredths of each other, which was checked); for Flint, Gemma 4 and
Lightning also KL over 300 tokens fed llama.cpp's own (their results folders). The server answered a chat, made a tool
call (whole and streamed) and answered after the tool's result; Lightning and Gemma 4 only on the PC.

## Against llama.cpp

- **Drafting rows (\*):** llama.cpp drafts with the same MTP layer (`llama-server --spec-type draft-mtp`, b11724), shown
  at its best number of drafted tokens (2-4); MoMoE drafts 4. Both sides: greedy, the same prompt, 400 tokens (MoMoE:
  its engine, the answer's second half; llama.cpp: its server's own timing over all 400). On the M1 Ultra, without
  drafting the two are level (92.0-96.1 tok/s against 89.6-96.1); llama.cpp gains more from drafting (Qwen3.6: 1.66x
  against 1.41x). Ornith's MTP layer is a weak drafter in both (llama.cpp: 54% of its drafts kept at 2, 28% at 4):
  without drafting both write 96.1. On the M4 Air, where it runs from a 14 GiB pool and the SSD, drafting still wins
  (30.7-31.7 against 24.6-25.2 tok/s without: a window shares the experts' reads).
- **The M4 Air** (2026-10-21, every expert in memory): MoMoE is ahead on the Qwen3.6 family, without drafting
  (Qwen3.6 33.2 against 28.9, Ornith 40.9 against 35.3) and more with it; level on LFM2.5-24B. Ornith's draft layer
  gains llama.cpp little there (36.1 with 2 drafted, slower with 3-4); MoMoE drafts only while that layer is sure (1.91
  tokens a window) and writes 49.3. Two runs each, within 0.4 tok/s.
- **llama.cpp's MoE cache** (`--moe-cache-mib`, added 2026-10-07 in llama.cpp #29887: the experts stay in RAM, a GPU
  cache holds the recently used ones; batches up to 32 tokens). Measured on the PC 2026-10-22 with a37cc46, its server
  at 32K context, greedy, the same 300-token chat as MoMoE's server. Flash-Next IQ3_XXS: its default fit 22.5 tok/s;
  every expert in RAM with a 4 / 5 GiB cache 21.2 / 23.1 (6 GiB and more do not fit beside its dense weights and KV
  cache); its fit plus a 2 / 4 GiB cache 18.3 / 17.1. Ornith-1.5-35B IQ4_XS: its fit 57.2; a 4 / 6 / 8 / 8.5 GiB cache
  45.3 / 62.6 / 77.8 / 75.1 (9 GiB does not fit); the 8 GiB cache with MTP drafting 63.5 (2 drafted; 3 does not fit).
  MoMoE's servers on the same chat, drafting as setup writes: 65.5 / 76.7 and 131.6 / 153.5 (two runs; 85% and 89% of
  the drafts kept).
- **GLM-5.3-Flash UD-IQ3_XXS on the Spark** (2026-10-22, llama.cpp a37cc46, `llama-bench` tg64 under a watchdog that
  kills it under 8 GB of available memory). Every layer on the GPU does not fit beside the system (its 112 GiB are RAM
  on the Spark: killed at 7 GB available), nor every expert on the CPU with llama.cpp's defaults (its host buffer and
  repacked copies are RAM too). With `--no-host 1 --repack 0 -lm mmap` the CPU's experts stay in the mapped file:
  `--n-cpu-moe` 99 / 26 / 16 / 10 / 6 / 5: 3.27 / 3.74 / 5.08 / 7.10 / 9.73 / 10.85 tok/s, the lowest available memory
  108 / 62 / 38 / 23 / 13 / 11 GB (4 would leave about 8). The table's earlier 1.3 was a 48-token check with every
  expert on the CPU, cold. UD-IQ4_XS's 2.3 / 2.5 there is from the same older check, not a placement sweep.
- **Flint** (2026-10-22; llama.cpp a37cc46, the commit that added it; 32K context; greedy 300 tokens, two runs): MoMoE
  runs it as LFM2.5 with every block attention, attention sinks, a 513-token sliding window on 35 of its 49 blocks
  (their keys and values in a ring of 1,792 positions: 0.14 GiB, not 2.3 at 32K), a shared expert, and llama.cpp's
  per-expert scaling of the down projection's input (left out on Metal for a few tokens, where its matrix-vector
  kernels read the input as f32: 38.4 -> 42.2 tok/s on the M4 Air, the same tokens). On the PC the CPU computes the
  experts the card does not hold, through the mailboxes, in a decode graph kept and replayed as a CUDA graph
  (UD-Q4_K_XL: 35 with the copies -> 57 -> 66 tok/s). Against llama.cpp over 300 tokens fed its own: KL means
  0.0015-0.0023, the same top token 98-100% (a 6K-token prompt: 0.008, with the ring or without it).
- **Gemma 4 and Lightning** (2026-10-22; llama.cpp a37cc46; greedy 300 tokens, two runs; llama.cpp `llama-bench`
  tg128 at its best fit, and for Gemma 4 UD-IQ3_XXS also its server at a 32K context, where the whole model no longer
  fits the 12 GB card): on the PC the CPU computes the experts the card does not hold; elsewhere every expert is in
  memory. Against llama.cpp over 299 tokens fed its own: Gemma 4 KL means 0.0005-0.0008 (all 299 greedy tokens the same
  on the PC and the Spark), Lightning 0.0048-0.0078 with the same top token 98-99% (as much without the CPU's experts:
  llama.cpp's reference computed 45 of its MoE layers on its CPU).
  [bench/results/2026-10-22-gemma4-lightning](../bench/results/2026-10-22-gemma4-lightning/README.md).

## Drafting per model

\* in the tables: the model's own multi-token-prediction (MTP) layer proposes up to 3 or 4 tokens ahead and the model
checks them in one pass, keeping only what it would have written itself. Without \*: the Flash-Next family on the
Spark, and Flash-Next IQ3_XXS and Swift on the M1 Ultra (where every expert fits only without the draft layer, which
measured faster) were measured without it, LFM has no MTP layer, and MoMoE does not yet draft with Bolt's.

Ornith-122B drafts only where (nearly) every expert is in memory, and setup adds `--mtp 3` only there (the Spark 26.4
-> 33.2 tok/s, the M1 Ultra 16.4 -> 19.8): on the PC the CPU computes the experts the card does not hold and drafting
was slower (19.1 -> 12.5), on the M4 Air (an 11 GiB pool, the rest from the SSD) 4.1-4.5 -> 4.0. Its greedy tokens
against llama.cpp's (64): the same on both Macs and on the Spark with drafting; `momoe-cuda` without drafting parts at
token 33, a three-way choice where each engine puts the weights differently (llama.cpp 0.53 / 0.28 / 0.19,
`momoe-metal` 0.56 / 0.39 / 0.05, `momoe-cuda` 0.36 / 0.38 / 0.25). Over 300 tokens fed llama.cpp's own, both are as
close to it (KL from llama.cpp's top 20, mean / median: `momoe-cuda` on the Spark 0.0107 / 0.0006, `momoe-metal` on the
M4 Air 0.0108 / 0.0006; the same top token 98.3% and 99.0%; Ornith-35B: 0.0028 and 0.0030).

GLM on the M4 Air drafts no tokens: from a pool of 9% of its experts drafting measured slower (1.0-1.1 against 1.4
tok/s).

## What the engines do for speed

- **A wait-for-event thread (`momoe-metal`).** One thread waits on the CPU's wait-for-event instruction while a request
  runs. Without it, macOS held the engine's main thread on the efficiency cores and the GPU's clock low, at random from
  run to run and on every request through the server (LFM2.5-8B on the M1 Ultra, through `momoe start`: 64-69 tok/s;
  with it 120-128). Thread priorities and launchd's `Interactive` did not change it.
- **Kept graphs (`momoe-metal`).** A graph is built once for its shapes (the tokens, the attention window, the outputs
  read back, the k-pool blocks a batch finishes) and computed again with the next token's inputs, the KV caches written
  at rows given as an input. Building it each token kept the GPU waiting (LFM2.5-8B: 0.83 ms of 8.3), and the GPU's
  clock follows how soon the next token's work arrives: llama.cpp with one thread, or without polling, wrote 78 tok/s at
  ~800 MHz; with two polling threads 182 at ~1,290 MHz. With graphs kept, LFM2.5-8B went from 120 to 185 tok/s (its
  server: 177.5) and LFM2.5-24B from 57 to 104; the other models 7-20% faster; the same tokens as before on every
  model. Only where every expert is in memory: with an expert pool (Unsloth's Flash-Next, GLM) a graph is built each
  token, as before. `MOMOE_NO_GRAPH_REUSE=1` turns it off.
- **In-place delta-net state and a direct router choice (`momoe-metal`),** from comparing its graph with llama.cpp's
  node by node (Qwen3.6 without drafting, 76.5 -> 92.2 tok/s, llama.cpp 90.3): the delta net's new state is written into
  its cache as one row, the shape ggml-metal's GATED_DELTA_NET + CPY fusion takes, so the kernel writes it in place (it
  was a 2 MB copy after each of the 30 layers: -1.8 ms a token); and with every expert in memory, the router's choice
  goes to the experts as it is (no copy of it, no slot lookup: 80 small kernels a token fewer). The same tokens as
  before on every model but the Coder, which parts at token 263: with 10 experts a token (no fused reduce for more than
  8) ggml-metal adds them in another order, and on this prompt fewer of its drafts were kept (48.9 tok/s, before 56.0;
  on two other prompts faster: 40.8 -> 46.8, 36.8 -> 37.4). `MOMOE_NO_DIRECT=1` turns the second off. On `momoe-cuda`
  only the first (no faster there: CUDA graphs).
- **CUDA graphs (`momoe-cuda`).** Kept graphs run as CUDA graphs (their kernels launched together, not one by one;
  graphs kept without it: 283): LFM2.5-8B on the PC 255 -> 305.5 tok/s (llama.cpp 305.8), on the Jetson with every
  expert in memory 18.2 -> 20.3 (llama.cpp 20.5), on the Spark 5-9% faster. Not for Bolt with its experts on the CPU
  (its own kept decode graph) nor across two cards (not measured).
- **Drafting's embedding lookup (`momoe-cuda`).** Drafting copied the whole token-embedding table from RAM to the GPU
  for every drafted token (540 MB for Qwen3.6, ~10 ms on the Spark); its rows are now looked up on the CPU, as the
  model's own pass did: on the Spark, Qwen3.6 59.5 -> 125.2 tok/s, KAT-Coder 57.5 -> 99.4, Ornith 66.2 -> 104.1. The
  same tokens as before on every model. GLM-5.3 drafts without that copy: 19.4 -> 19.7.
- **The pool fitted to the GPU's working set (`momoe-metal`).** At load the largest graph's buffers are reserved and
  the expert pool shrinks to what the working set leaves less 512 MiB (an 8 GB M1: Gemma 4 with setup's 3 GiB pool
  needed 6.45 of 5.33 GiB and crashed). `MOMOE_METAL_MARGIN_MIB`, `MOMOE_NO_METAL_FIT=1`.

## The two-card measurements

The helper tier with a 262K context: the KV cache leaves the 4070 Ti room for fewer experts (1,071 instead of about
2,850). In everyday use with that config (2026-10-18, 7 answers of 115-4,572 tokens), it wrote 43.6-56.1 tok/s with a
54-71% hit rate in the main card's cache. The first answer after each start was about 20 tok/s, while the caches fill.

Ornith-1.5-122B UD-IQ3_XXS on the same two cards (2026-10-21, `momoe-cuda`): with `--helper-tier` the CPU computes the
experts neither card holds, as on one card, and the 3060 keeps its share out of RAM. It writes 12.0-12.2 tok/s against
19-21 on the 4070 Ti alone (each layer's hand-off between the cards costs ~27 ms a token), and against 8.7 with the
helper and the experts copied in instead.

The layer split runs neither with the CPU's experts nor with the mailboxes; setup splits GLM-5.3-Flash by itself
across two cards under 16 GB each. Not yet measured for the tables.
