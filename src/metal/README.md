# momoe-metal (work in progress)

Momoe's engine for Apple Silicon: qwen35moe models (Qwen3.6-35B-A3B, Ornith-1.5, KAT-Coder) and qwen4exp
(Qwen3.8-Flash-Next) with the routed experts read from the GGUF into a pool as tokens need them, so a model larger than
the Mac's free memory can run.

## Try it (Ornith-1.5 on an Apple Silicon Mac)

Needs Xcode's command line tools (`xcode-select --install`), Python 3.10 or newer (macOS's own python3 is 3.9:
`brew install python`), 16 GB of memory or more and ~25 GB of disk. From the repository's root:

```sh
./momoe setup              # builds momoe-metal, downloads Ornith-1.5 IQ4_XS (19.3 GB), writes config/ornith-mac.json, starts it
./momoe run claude         # or codex, opencode, lmstudio
./momoe stop
```

Setup picks the expert pool from the Mac's memory: every expert in memory from ~28 GB, else the memory less 11 GiB
(16 GB: a 5 GiB pool and a 32K context; measured on the M4 with those settings: 11.2 tok/s, the server 8.6 GB). It
prints a speed test command at the end. Flash-Next is set up by hand (a 75 GB download, and its MTP layer packed from
the checkpoint): design/mac-port.md.

## Build and run

```sh
src/metal/build.sh          # patches src/metal/ggml (patch_ggml_pause.py), builds it and the engine into build-metal/
build-metal/momoe-metal -m models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf --tokens 760,6511,314,9338,369 -n 64 \
    [--pool-gib 7 | --slots N] [--nocache] [--preload] [--ctx 4096]
```

- A split model (Flash-Next: two files) is opened by its first file (`-00001-of-0000N.gguf`).
- `--pool-gib`: the expert pool's size (default: every expert). `--nocache`: expert reads bypass macOS's file cache
  (F_NOCACHE), as on a Mac without the RAM to cache the file. `--preload`: every expert loaded at the start; the graphs then run without pauses. A pool that
  holds every expert (the default without `--pool-gib`) preloads by itself (`MOMOE_NO_PRELOAD=1`: pauses as before).
- `MOMOE_DEBUG=1`: tensor sums, comparable with llama.cpp's `llama-eval-callback`. `MOMOE_READERS=N`: reader threads
  (default 8). `MOMOE_NO_PAUSE=1`: the scheduler-callback path instead of pause points.
- `--batch N`: prompt chunk size (default 512). `MOMOE_TOP=1`: the 5 most likely tokens at each step.
  `MOMOE_NO_QSA=1`: Flash-Next's attention to every position (a test).

## How it works

- The dense weights (1.56 GiB for Ornith IQ4_XS) are loaded whole into a Metal buffer (shared memory).
- The routed experts live in a pool: one tensor each for gate, up and down, `[ne0, ne1, n_slots]`. A layer's expert
  matmuls read the pool through `get_rows(slot_map, chosen_experts)`: the per-layer expert -> slot table, looked up
  on the GPU (a real dependency on the router's choice; slot numbers in a separate tensor were read by Metal's
  reordered graph before they were written).
- One command buffer per token. After each layer's router (`topk-N`), the GPU signals a shared event and waits
  (`patch_ggml_pause.py`, in Momoe's copy of ggml); the engine makes the chosen experts resident (missing ones read
  first, by 8 reader threads, straight into the pool) and lets the GPU go on.
- Prediction: the same top-k also routes the next layer's router over this layer's hidden state; after releasing the
  GPU the engine reads those experts in the background (copied out first: after the release the allocator reuses the
  tensor's memory). `MOMOE_PREDICT2=1` also predicts two layers ahead (as fast, a third more read).
- Replacement: the least used expert (use counts halved every 64 tokens), recency breaking ties; predictions waiting
  for their layer are not replaced. `MOMOE_POLICY=lru`: least recently used. `--profile P` pins Strata's profile's
  hottest experts (`MOMOE_PIN_FRAC`, default 0.6): slower here, the profile did not match the workload.
- Why waiting matters twice: while the GPU waits at a pause, macOS lowers its clock (340-610 MHz measured during SSD
  waits, ~1,465 MHz when busy), so the next layers also run slower.
- The graph follows llama.cpp's qwen35moe and qwen4exp op for op.
- Experts of different formats (Strata's Flash-Next files quantize them per layer: 10 combinations in IQ3_XXS) live in
  one group of slots per format; each group gets the same share of its experts, at least 16 tokens' worth.  A prompt
  chunk reads a missing expert into the pool while slots are free, else straight into the staging copy.

## Flash-Next (qwen4exp)

- Hyper-connections (4 streams; llama.cpp's fused hc_pre_gated / hc_post ops, on Metal), the delta net's sigmoid
  output gate, the final mixer as the output norm.
- PLE (the n-gram embedding at layer 1): each token's 16 rows of the 28.8 GB table (in the second file) are read on
  the CPU (pread + IQ4_NL to f32) and fed to the graph; the hash is llama.cpp's (EOS cuts the n-gram, positions
  before the start read as EOS); the convolution's history is part of the recurrent state (saved / restored / reset
  with it).
- QSA (sparse attention in the 12 attention layers): each layer keeps its indexer keys per position and a pooled key
  per block of 4 positions (mean, normed, rotated to the block's first position: pooled when the block is complete);
  past 2,048 positions a token attends to its 512 best scoring blocks and its unfinished block (llama.cpp's
  build_qsa_sel, ported op for op; every op on Metal: a graph split would stop the pauses). Up to 2,048 positions
  every block is kept: plain attention, as in llama.cpp.
- The MTP layer is not in llama.cpp's GGUFs: `--mtp N --mtp-gguf FILE` loads it from the checkpoint's `mtp.*`
  tensors as `tools/mtp_pack.py --src mtp --experts q4_0 --out mtp/mtp-q4_0.gguf` packs them (1.48 GiB; Strata's
  Q2_0 experts have no Metal kernel). The graph is llama.cpp's qwen4exp `graph_mtp` (each hc stream of the hidden
  state normalized on its own, eh_proj per stream, the layer's own final mixer before the head) with dense attention,
  as Strata's drafter; the stored norms are zero-centred and get +1, as llama.cpp's converter does for the model's
  own. The hidden state it reads is the four streams before the final mixer (10,240 values per position).

## Measured (M4 MacBook Air, 32 GB, macOS 27, Ornith-1.5-35B-A3B IQ4_XS, 2026-10-12)

Correctness: two prompts, 32 greedy tokens each, identical to llama.cpp (b11429) with every expert in the pool and
with a 7 GiB pool reading from the SSD, with every policy and prediction setting below; layer sums match
llama-eval-callback to 5-6 digits.

Writing, tok/s, second half of a 256-token answer; "SSD": `--nocache` after `sudo purge`:

| | tok/s |
|---|---|
| llama.cpp, all in memory | 33.7 |
| every expert preloaded, no pauses | 29.6 |
| preloaded, 40 pauses per token | 26.3 (~110 us per pause; a scheduler stop per layer cost ~520 us) |
| 7 GiB pool (16 GB Mac), SSD, LRU, no prediction, 1 reader | 7.8-9.0 |
| 7 GiB, 8 readers | 11.9 |
| 7 GiB, 8 readers, predicting 1 layer ahead (LRU) | 10.7 |
| 7 GiB, profile pinned 50% / 80% | 9.7 / 9.0 |
| 7 GiB, least used + prediction (now the default) | 13.5 (hit rate 94.4%; 25 GB read for 256 tokens) |
| 14 GiB pool (24 GB Mac), SSD, least used + prediction | 23.5 (hit rate 98.1%) |

Drafting with the MTP layer (`--mtp 4 --draft-vocab data/draft_vocab_en.bin`, `--spec-min-p 0.5`), code prompt, 256
tokens: 1.62 tokens per window, 96% of drafts accepted; output identical to plain greedy (also with min p 0, where most
drafts fail and windows are run again).

| | without drafting | with drafting |
|---|---|---|
| every expert in memory | 25.5 | 30.8 |
| 7 GiB pool, SSD | 12.1 | 13.6 |

- A window of T tokens costs 40 ms + ~11 ms per extra token (in memory, with pauses): the 40 pauses are paid once.
- The MTP layer's experts are Q4_0 (the main ones IQ4_XS): loaded whole (0.45 GB), outside the pool. Its head runs
  over Momoe's draft vocabulary (the output head's rows for 40K tokens: 2.9 ms per window, 7-12 ms with the full
  248K head).
- A failed draft: the delta net's state is copied aside before each window with drafts (~1 ms) and the accepted
  tokens are run again (4 of 157 windows here). ggml_gated_delta_net's per-token snapshots (K = T) cost 19-30 ms per
  window on the M4: slower than running a window again now and then.
- llama.cpp's MTP on this Mac and file: 13-34% of drafts accepted with no threshold (19.8 tok/s vs 33.7 without).

## Serving (`--serve`): momoe start / momoe run on macOS

`momoe-metal --serve` speaks the engine protocol of Momoe's server (serve/server.py) for one conversation at a time:
INFO/READY at start; GEN (the whole prompt each time) answered with RESUME, PP, T per token and DONE; STOP at any time
(stdin on its own thread; a stale one between requests ignored); VRAM answered ERR; QUIT or end of input exits.

- Prompt reuse: the state continues when a prompt extends what it holds; else the checkpoint saved at the end of the
  previous prompt is put back when the prompt starts with it (an agent's next turn); else from the start (the delta
  net's state cannot be rewound to any position).
- Greedy requests draft with the MTP layer; with a temperature the token is sampled on the CPU from the last
  position's logits (temperature, top_k, top_p, min_p, seed), without drafting.
- Prompt chunks of 32+ tokens read a layer's experts from a staging copy (`STAGE_MIN_TOKENS`): Metal's batched
  expert matmul takes at most 1,024 experts per tensor, the pool has more.
- The tokenizer is the server's (the pack's `tokenizer/` folder, copied from a Linux pack for now).

Tried on the M4 with `config/ornith-mac.json` (all experts in memory, `--ctx 65536`, `"fit_max_tokens": true`):
`momoe start ornith-mac` reported ready; OpenAI chat (17*23 -> 391), streaming, Anthropic /v1/messages (12*12 ->
144); `momoe run claude -p "What is 17*23? ..."` answered 391 (Claude Code's 15,998-token prompt read in 68 s,
236 tok/s; 9.8 tok/s writing at that length). Protocol by hand: the first GEN's tokens identical to llama.cpp; a
second GEN extending the conversation reused 37 tokens; a sampled GEN; VRAM; QUIT. A 1,103-token prompt (two staged
chunks): 16 tokens identical to llama.cpp.

## Flash-Next measured (M4 MacBook Air 32 GB, Qwen3.8-Flash-Next IQ3_XXS, 2026-10-12)

Against llama.cpp b11429 on the CPU (`-ngl 0`; on Metal it runs out of memory):

- Short and code prompts, 32 greedy tokens: identical, with 12, 6 and 3 GiB pools, from the file cache and from the
  SSD.
- 1,100 tokens: 8 tokens identical, then a step where both rank the same two tokens with the order swapped (here
  0.66 / 0.33, llama.cpp 0.36 / 0.63); our own chunk sizes move that step as much (512: 0.66, 31: 0.80, 1: 0.42 and
  llama.cpp's token). Other steps within ~0.05.
- 2,600 tokens: the first step is a near tie, swapped (0.33 / 0.21 vs 0.25 / 0.28).
- 8,015 tokens (QSA keeps 512 of 2,003 blocks): the same top 5 tokens in the same order as llama.cpp (0.62 / 0.20 vs
  0.42 / 0.26); without QSA the order differs (0.75 / 0.13).

Speed, IQ3_XXS (3.83 GiB dense, 39.97 GiB of experts):

| pool | writing (2nd half) | prompt |
|---|---|---|
| llama.cpp, CPU | 4.2 tok/s | 8 tok/s |
| 12 GiB, file cache | 9.3-9.6 tok/s | 56-62 tok/s (1,100-8,015 tokens) |
| 12 GiB, SSD (`--nocache`) | 10.5 tok/s | |
| 6 GiB, SSD | 5.1 tok/s | |
| 3.4 GiB, SSD | 4.1 tok/s | |

Short runs (32 tokens), no `sudo purge`, the file cache partly warm: rough figures.  Serving
(`config/flashnext-mac.json`, 14 GiB pool, `--ctx 32768`): OpenAI 17*23 -> 391, streaming, Anthropic 12*12 -> 144;
`momoe run claude -p` -> 391, its 15,052-token prompt in 303 s (49.6 tok/s), 6.2 tok/s writing (with QSA; the same
before QSA). The startup banner still describes a PC with a graphics card.

Not explained yet: at 8,015 tokens our chunk sizes agree (0.62-0.64 for the top token) but llama.cpp's CPU run gives
0.42. Its CPU matmuls round activations to 8 bits, ours do not; a Metal reference is not possible here (llama.cpp
runs out of Metal memory with Flash-Next, also with every expert on the CPU).

Drafting with the MTP layer (2026-10-13; 12 GiB pool, code prompt, 200 tokens, two rounds with 90 s pauses between
runs: the Air throttles under a long load):

| | writing | tokens per window |
|---|---|---|
| no drafting | 6.8 / 7.3 tok/s | 1 |
| `--mtp 3 --mtp-gguf mtp/mtp-q4_0.gguf` | 8.9 / 8.6 tok/s | 3.26 (138 of 166 drafts) |
| `--mtp 2` | 8.7 / 8.6 tok/s | 2.64 |
| `--mtp 3` with `--draft-vocab data/draft_vocab_en.bin` | 7.8 / 8.0 tok/s | 3.12 |

Short and code prompts, 32 tokens: identical to llama.cpp with drafting, from the CLI and from `--serve`. A window of
4 tokens costs ~315 ms against ~126 ms for one (4 tokens route to up to 40 experts a layer: more reads and more
misses), and a failed draft runs the accepted tokens again (~34 ms per window on average).

## No-stall mode (`--nostall sub|drop`, opt-in, changes the output; 2026-10-13)

The idea of virtual texturing: a token's graph runs without pauses. Each layer routes among the experts already in
the pool (`sub`: the best ones there; `drop`: its own choice, the missing ones weighted 0) and reports what it
wanted; between graphs the CPU reads those and the predictions into the pool and publishes them (masks, weights,
slot tables) once read, so nothing changes under the GPU. The prompt, and any graph while some layer has fewer than
4 x n_used experts in the pool, take the exact path with pauses. A missing expert's slot points at a slot of zeros.

Teacher-forced on the exact path's own tokens, 300 tokens (`MOMOE_KL_DUMP=FILE` writes a run's tokens and log
probabilities, `MOMOE_KL_REF=FILE` feeds them to another run and reports KL(exact || this) per step;
baseline/run/kltest.sh on zephyr). The exact path against itself: KL 0.0000 (max 0.0003), top-1 100%.

| | missing | KL mean | median | p99 | top-1 | tok/s exact -> no-stall |
|---|---|---|---|---|---|---|
| Flash-Next 12 GiB, code, sub | 13.6% | 0.116 | 0.005 | 3.07 | 94.3% | 7.6 -> 9.7 |
| Flash-Next 12 GiB, code, drop | 13.7% | 0.120 | 0.005 | 2.34 | 92.7% | 7.6 -> 9.7 |
| Flash-Next 12 GiB, 1,100-token document, sub | 10.8% | 0.018 | 0.0005 | 0.26 | 96.3% | 8.4 -> 8.9 |
| Flash-Next 12 GiB, document, drop | 10.9% | 0.022 | 0.0007 | 0.40 | 97.0% | 8.4 -> 8.8 |
| Ornith 7 GiB, code, sub | 12.7% | 0.021 | 0.0004 | 0.26 | 97.7% | 16.5 -> 21.0 |
| Ornith 7 GiB, code, drop | 13.1% | 0.027 | 0.0007 | 0.43 | 97.3% | 16.5 -> 19.9 |

On the Mac Studio M1 Ultra (en, 64 GB), where a pause costs ~550-850 us, the same tests:

| | missing | KL mean | median | p99 | top-1 | tok/s exact -> no-stall |
|---|---|---|---|---|---|---|
| Flash-Next 34 GiB, code, sub | 7.0% | 0.043 | 0.0013 | 1.19 | 93.3% | 11.8 -> 19.8 |
| Flash-Next 34 GiB, code, drop | 7.0% | 0.042 | 0.0016 | 0.93 | 94.0% | 11.8 -> 20.3 |
| Flash-Next 34 GiB, document, sub | 0.3% | 0.0016 | 0.0001 | 0.015 | 98.0% | 12.3 -> 20.9 |
| Flash-Next 34 GiB, document, drop | 0.3% | 0.0021 | 0.0001 | 0.029 | 98.7% | 12.3 -> 21.4 |
| Flash-Next 20 GiB, code, sub | 8.3% | 0.048 | 0.0019 | 0.73 | 92.7% | 12.0 -> 20.6 |
| Ornith 7 GiB, code, sub | 12.8% | 0.014 | 0.0002 | 0.21 | 98.3% | 22.8 -> 38.2 |
| Ornith 7 GiB, code, drop | 12.9% | 0.015 | 0.0003 | 0.21 | 97.3% | 22.8 -> 39.3 |

The exact path is held back by the pauses there (12 tok/s at 20 GiB and at 34 GiB; with 99.8% hits on the document,
still 12.3), so no-stall gains ~70%. With drafting (34 GiB, code, 200 tokens): 16.4 -> 31.7 tok/s (not like for like:
the answers differ, see below). llama.cpp with the whole model in memory: 34.0 tok/s. The exact path still matches
llama.cpp there (Flash-Next 34 GiB with and without drafting, Ornith 7 GiB: 32 of 32).

A pause every N layers (`--nostall sub --nostall-every N`): layers 0, N, 2N... keep their pause and
are exact; at each pause the CPU reads what the layers since the last one wanted, starts the predicted reads, and
publishes what has been read for the layers after it (every slot taken before publishing, so nothing a published
layer reads changes). `--nostall-every 1` gives the exact path's tokens (32 of 32, Ornith and Flash-Next). zephyr, Flash-Next
12 GiB, 300 tokens teacher-forced, 60 s pauses between runs ("missing": of the unpaused layers' wanted experts):

| | code: KL mean / p99 / top-1 | tok/s | document: KL mean / p99 / top-1 | tok/s |
|---|---|---|---|---|
| exact | | 7.8 | | 9.1 |
| every 2 (5.1% missing) | 0.012 / 0.24 / 98.0% | 6.9 | 0.0071 / 0.092 / 98.0% | 8.7 |
| every 4 (11.1%, 9.2%) | 0.055 / 2.27 / 97.0% | 9.0 | 0.0148 / 0.34 / 97.3% | 9.7 |
| every 8 (12.7%, 10.3%) | 0.067 / 2.48 / 94.3% | 9.7 | 0.0123 / 0.18 / 97.3% | 10.2 |
| no pauses (13.6%, 10.9%) | 0.119 / 4.05 / 93.3% | 9.9 | 0.0184 / 0.34 / 97.0% | 10.1 |

Every 2 is slower than exact here (each pause also publishes the tables of every later layer, 512 experts each);
on the M4 a pause is cheap, so the gain is in the SSD waits taken off the unpaused layers.

en (M1 Ultra, pauses cost 0.5-0.9 ms), same runs without cooling pauses; `--nostall-every 1` matched llama.cpp there
too (32 of 32, Flash-Next 34 GiB and Ornith 7 GiB):

| | FN 34 GiB code: KL / p99 / top-1 | tok/s | FN 34 GiB document | tok/s | FN 20 GiB code | tok/s | Ornith 7 GiB code | tok/s |
|---|---|---|---|---|---|---|---|---|
| exact | | 11.8 | | 12.3 | | 12.1 | | 22.8 |
| every 2 | 0.0077 / 0.19 / 98.0% | 12.8 | 0.0009 / 0.011 / 99.0% | 15.3 | 0.0106 / 0.16 / 97.3% | 13.0 | 0.0062 / 0.11 / 99.0% | 23.8 |
| every 4 | 0.0231 / 0.39 / 95.3% | 14.7 | 0.0017 / 0.026 / 98.3% | 15.9 | 0.0307 / 0.42 / 95.0% | 15.3 | 0.0093 / 0.20 / 98.7% | 26.9 |
| every 8 | 0.0221 / 0.36 / 95.0% | 16.1 | 0.0013 / 0.017 / 98.7% | 17.2 | 0.0241 / 0.41 / 95.3% | 16.2 | 0.0122 / 0.25 / 98.0% | 30.0 |
| no pauses | 0.0434 / 1.20 / 93.3% | 19.8 | 0.0016 / 0.015 / 98.0% | 20.9 | 0.0509 / 0.77 / 93.3% | 20.6 | 0.0127 / 0.24 / 98.0% | 38.2 |

Missing (of the unpaused layers' wanted experts), every 2 / 4 / 8 / none: FN 34 code 3.1 / 5.8 / 6.6 / 7.0%, FN 20
code 3.3 / 6.7 / 7.6 / 8.2%, document 0.2-0.3%, Ornith 5.3 / 10.4 / 11.8 / 12.8%. On en, Flash-Next code: every 2 has
a sixth of the no-pause KL but keeps 1.0 of its 8.0 tok/s gain; every 8 has half the KL and keeps 4.3.

About 2.5x as many wanted experts are missing as the exact path misses (5%): the reads start only after the whole
token, so the next-layer predictions no longer hide them. With drafting (`--mtp 3`, 12 GiB, code, 200 tokens, 90 s
pauses): 8.8-9.0 -> 15.1-15.2 tok/s, but the answers differ after 3 tokens and no-stall's (memoized and DP variants
of the function) drafts better (3.81 tokens per window against 3.26): that comparison is not like for like.

## Next

- Flash-Next: speed (~100 ms per token with the experts in memory, about 35 ms of weight
  reads: the hyper-connection matrices are BF16, 1.3 GB per token, and the predictions add 0.6 GB); the 8K gap above;
  a run on the 64 GB Mac Studio.

1. Prompt speed: 236 tok/s on 16K tokens (llama.cpp 270-333); every chunk stages all 256 experts of each layer.
2. Writing at long context (9.8 tok/s at 16K).
3. Setup on macOS: `momoe setup` making these configs, building momoe-metal, the tokenizer folder without a Linux
   pack; setup's messages about NVIDIA and RAM.
4. Still waiting ~14 ms per token at 7 GiB (predictions late or wrong).
