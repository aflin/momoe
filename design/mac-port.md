# Momoe on Apple Silicon: the Metal engine

**Status:** in development (2026-10-12), uncommitted on the Mac (`src/metal/`). Built and measured on one Mac only.

## The goal

Run mixture-of-experts models on Macs whose memory cannot hold them, by keeping the dense weights in memory and
reading the routed experts from the SSD as tokens need them, as Strata does with a graphics card and system RAM on a
PC. First target: Qwen3.6-35B-A3B and its fine-tunes (Ornith 1.5, KAT-Coder) on 16-24 GB Macs. Then
Qwen3.8-Flash-Next (runs on the 32 GB MacBook since 2026-10-12; the 64 GB Mac Studio next).

Not a wrapper around llama.cpp: llama.cpp already runs these models on a Mac whose memory holds them, and on Metal it
needs the whole model in GPU-usable memory (it cannot run Ornith on the GPU of a 16 GB Mac). The engine is Momoe's
own; it uses ggml (llama.cpp's MIT-licensed kernel library) for the Metal kernels, and follows llama.cpp's qwen35moe
graph op for op so its results can be checked against llama.cpp.

## The test machine

MacBook Air, Apple M4 (10-core GPU), 32 GB, macOS 27, 2 TB SSD (zephyr). Fanless.

## llama.cpp on this Mac (b11429, the starting point)

| | writing | reading a prompt |
|---|---|---|
| Ornith-1.5-35B-A3B IQ4_XS (18 GB file), all on the GPU | 33.7 tok/s | 333 tok/s (5.6K tokens), 270 tok/s (20K) |
| the same with its MTP draft layer (`--spec-type draft-mtp`) | 19.8 tok/s | 296 / 239 tok/s |
| Qwen3.8-Flash-Next IQ3_XXS (75 GB), on the GPU | fails: Metal out of memory (it maps the whole file to the GPU) | |
| the same, CPU only, read from the SSD as needed | 4.2 tok/s | 8 tok/s (a 20K prompt: 45 min) |

llama.cpp's draft layer accepted 13-34% of its drafts on these prompts.

## Why it can work: the numbers

- Ornith IQ4_XS: 1.58 GiB of dense weights (needed for every token) and 16.36 GiB of routed experts: 256 per layer x 40
  layers, 1.6 MiB each; a token uses 8 per layer, 320 reads (512 MiB) if none is cached.
- The SSD, random 1.6 MiB reads with the file cache cleared (`sudo purge`, F_NOCACHE): ~4 GB/s one at a time, ~8 GB/s
  with 4 in flight (one earlier run measured 2.4 GB/s for single reads).
- On Linux, an expert cache holding 31% of Ornith's experts hit 72-85% in real sessions.
- On Apple Silicon the CPU and the GPU share memory: the expert pool is one buffer the GPU reads, and an SSD read
  lands where the GPU uses it, with none of the copies to a card that Strata manages on a PC.

## How the engine works

- **Dense weights** (1.56 GiB) are read whole into a Metal buffer.
- **The expert pool**: one tensor each for gate, up and down, `[ne0, ne1, n_slots]`. A layer's expert matmuls read it
  through `get_rows(slot_map, chosen_experts)`: a per-layer expert -> slot table looked up on the GPU.
- **Pause points**: one command buffer per token. After each layer's router, the GPU signals a Metal shared event and
  waits; the engine reads the chosen experts from shared memory, makes the missing ones resident, fills the table and
  signals back. This needs a small patch to ggml's Metal backend (`src/metal/patch_ggml_pause.py`, applied to Momoe's
  copy of ggml in `src/metal/ggml`).
- **Readers**: 8 threads read experts from the GGUF straight into the pool (pread), the ones the GPU is waiting for
  first.
- **Prediction**: the same top-k step also routes the next layer's router over the current hidden state; after
  releasing the GPU the engine reads those experts in the background.
- **Replacement**: the least used expert (use counts halved every 64 tokens), recency breaking ties; predictions
  waiting for their layer are kept.
- **Drafting (MTP)**: the model's own draft layer (layer 40) proposes tokens while it is at least 50% sure; a window
  of [token + drafts] is checked in one pass. Its experts (Q4_0, the main ones are IQ4_XS) are loaded whole (0.45 GB);
  its output head runs over Momoe's draft vocabulary (40K tokens). A failed draft: the delta net's state is copied
  aside before each window with drafts and the accepted tokens are run again.
- **Prompt chunks of 32+ tokens** read a layer's experts from a staging copy of one layer, filled at the pause:
  Metal's batched expert matmul takes at most 1,024 experts per tensor, the pool has more.
- **Serving** (`--serve`): the engine protocol of Momoe's server, so `momoe start` and `momoe run` work: streaming,
  STOP, prompt reuse across an agent's turns (the state continues, or a checkpoint at the end of the last prompt is
  put back), greedy with drafting, sampling on the CPU.

## What was found on the way

- **A pause only on a miss is not possible this way (tried 2026-10-13).** The GPU cannot decide by itself whether to
  wait: a Metal wait on an event is encoded unconditionally, and a kernel that spins on shared memory never sees the
  CPU's write (nor the CPU the kernel's) while it runs; both versions hung at the first miss. What remains: no pauses
  at all when the pool holds every expert (now automatic). On the M4 a pause costs little (Ornith, every expert in
  memory, 128 tokens: 35.1 ms per token without pauses, 39.7 with, 36.8 with every wait released ahead); on the M1
  Ultra it costs ~550 us.
- **Polling a flag word instead of the event did not help (tried 2026-10-13).** The CPU already spins on the event
  (no sleeping thread to wake). A test switch had the GPU copy the first chosen expert into a per-layer word right
  after the router, with the CPU spinning on that word and still checking the event before releasing. On the M4
  (Ornith, every expert in memory, pauses forced on, code prompt, 128 tokens, 3 alternating runs): 41.5-41.9 ms per
  token against 40.9-41.7 for the event, 37.3 with no pauses. It also hung in 2 of 5 runs: the CPU never saw the
  write, the same visibility problem as above. Removed; the attempt is kept in momoe-notes as
  momoe_metal.cpp.flagpoll-attempt.
- **Computing the pool's experts while the misses are read did not help (tried 2026-10-13, MOMOE_OVERLAP).** Two
  pauses per layer: at the first the misses' reads start and they point at a slot of zeros, the GPU computes the
  experts in the pool, at the second the CPU waits for the reads and a second pass computes the missed ones (it has
  to depend on the first pass's output, or Metal's reordering runs it early on experts still being read). Flash-Next,
  12 GiB pool, 200 tokens: the read waits fell from 18.9 to 15.3 ms per token, but the second pass (all 10 experts
  again, mostly the zero slot) and the second pause added more: 7.3 -> 7.0 and 6.9 -> 5.6 tok/s, with drafting 8.9 ->
  7.8. Removed; kept in momoe-notes as momoe_metal.cpp.overlap-attempt.
- **Metal's IO queue reads no faster (2026-10-13).** MTLIOCommandQueue against 8 pread threads, expert-sized reads
  (1.75 MB) at random offsets of the Flash-Next file, `sudo purge` first, batches of 1 / 3 / 8 waited for:
  0.76 / 2.13 / 4.59 ms against 0.79 / 2.13 / 4.55 ms (2.2-3.1 GB/s). The SSD is the limit.
- **No-stall mode (`--nostall`, opt-in, 2026-10-13).** Metal 3.2's coherent(device) memory is coherent between
  threadgroups, not with the CPU mid-kernel, and Metal 4's queue-level events are still a round trip: a pause stays a
  pause. So, as virtual texturing does, the GPU never waits: each layer routes among the experts in the pool and
  reports what it wanted, the CPU reads those between tokens. Teacher-forced KL against the exact path, 300 tokens:
  Flash-Next code 0.116 (top-1 94.3%), a document 0.018 (96.3%), Ornith code 0.021 (97.7%); speed 7.6 -> 9.7,
  8.4 -> 8.9 and 16.5 -> 21.0 tok/s on the M4. On the M1 Ultra (en), where pauses dominate: Flash-Next 34 GiB
  11.8 -> 19.8 tok/s (code, KL 0.043) and 12.3 -> 20.9 (document, KL 0.0016), Ornith 7 GiB 22.8 -> 38.2 (KL 0.014).
  Numbers and how it works: src/metal/README.md.
- **Why a pause costs more on the M1 Ultra (2026-10-13).** momoe-notes/pausebench.mm: 48 layers of dummy dispatches
  in one command buffer, a shared-event signal and wait after each. Per pause, M4 / M1 Ultra: 53-110 / 216-241 us
  with the CPU releasing each wait as it sees the signal; 0-22 / 228-519 us with every wait released before the
  commit (no CPU round trip at all). On the M4 the cost is the round trip to the CPU; on the M1 Ultra the event
  signal and wait cost that much by themselves, so a faster CPU side cannot help there, only fewer pauses. Holding
  each pause 200 us cost 32-49 us extra on the M4 and 20-210 us on the M1 Ultra (1,225 with narrow work); a second
  queue keeping the GPU busy did not help either. momoe-metal's pauses on en (550-850 us) cost more than the
  benchmark's, not explained yet (more command buffers and encoders per layer in ggml, the CPU's work at the pause).
- **A pause through shared memory instead of events does not work (2026-10-13).** momoe-notes/flagbench.mm: one
  GPU thread at the end of each layer writes a word and spins on another that the CPU writes (the CUDA doorbell).
  The CPU never sees the GPU's write while the kernel runs, also with a device-scope fence (MSL 3.2), on the M4 and
  the M1 Ultra. With the arrival through an event and only the release through memory, the GPU did not see the CPU's
  write either: 4 pauses took 3.8 s (M4) and 4.4 s (M1 Ultra), the GPU giving up its bounded spin 2 and 4 times.
  Metal makes CPU writes visible to the GPU, and back, only at command boundaries; events are the only way in.
  Apple publishes nothing on what made events cheaper after the M1; the two machines also differ in macOS (27.0 on
  the M4, 15.6.1 on en) and in dies (one against two), which this benchmark cannot separate.
- **Metal reorders the graph.** With the slot numbers in a separate tensor, the expert matmuls had no dependency on
  the router's choice, and ggml's Metal backend ran them before the experts were loaded (layer 0's expert output was
  exactly 0). The slot lookup on the GPU (`get_rows` on the chosen experts) makes the dependency real.
- **Stopping the GPU is expensive.** Ending the command buffer at each layer (ggml's scheduler callback) cost ~520 us
  per stop, 20.7 ms per token. A pause inside the command buffer (shared event signal and wait) costs ~110 us.
- **A waiting GPU slows down.** While the GPU waits at a pause for SSD reads, macOS lowers its clock: 340-610 MHz
  measured during reads, ~1,465 MHz when busy, so the next layers run slower too. Waiting costs about twice its
  length; the fix is not to wait (prediction, parallel reads).
- **A race**: the predictions were read after releasing the GPU, which by then had reused that tensor's memory
  (garbage expert numbers, a crash in one configuration). They are copied out before the release.
- **Snapshots are slow here**: ggml_gated_delta_net's per-token state snapshots (for rolling back a window) cost
  19-30 ms per window on the M4; copying the state aside and running a window again after a failed draft is cheaper.
- **Strata's expert profile did not help**: pinning the profile's hottest experts (50% / 80% of the pool) was slower
  than the adaptive policy on these prompts (9.7 / 9.0 vs 12.1 tok/s).

## Correctness

Greedy output compared with llama.cpp (b11429) on the same file: two short prompts (32 tokens each) and a 1,103-token
prompt (16 tokens; two staged chunks): identical, with every expert in the pool, with a 7 GiB pool reading from the
SSD, with every replacement and prediction setting, and with drafting (also with no confidence threshold, where most
drafts fail). Every layer's output sums match llama.cpp's `llama-eval-callback` to 5-6 digits.

## Speed

Ornith-1.5-35B-A3B IQ4_XS, a code prompt (15 tokens), 256 tokens greedy, `sudo purge` before each run, all six in one
session. "SSD": expert reads bypass the file cache (F_NOCACHE). Writing speed, tok/s.

The whole run, from an empty pool:

| | without drafting | with drafting | gain |
|---|---|---|---|
| every expert in memory | 25.5 | 30.9 | +21% |
| 14 GiB pool (a 24 GB Mac), SSD | 18.6 | 22.1 | +19% |
| 7 GiB pool (a 16 GB Mac), SSD | 12.0 | 13.5 | +13% |

Warmed up (the second half of the answer, tokens 128-256):

| | without drafting | with drafting | gain |
|---|---|---|---|
| every expert in memory | 25.5 | 31.6 | +24% |
| 14 GiB pool (a 24 GB Mac), SSD | 23.7 | 29.5 | +24% |
| 7 GiB pool (a 16 GB Mac), SSD | 13.4 | 14.7 | +10% |

The second half is also different text from the first, so its drafting gain is partly the text, not only the warm
pool. With every expert in memory the whole run and the second half without drafting are the same (25.5): there is
nothing to warm up.

Drafting: 1.62 tokens per window, 98 of 102 drafts accepted (the threshold lets few drafts through on raw-text
prompts; agent work is likely more predictable, untested). A window of 2 tokens costs 1.3x a single token in memory
(51.2 vs 39.9 ms), 1.6x with the 7 GiB pool (131.3 vs 82.0 ms: the extra token's experts are more often on the SSD),
so the gain shrinks with the pool.

How the 16 GB and 24 GB figures were chosen, and what they do not show: the pool was sized to what such a Mac could
spare (16 GB - macOS and apps ~5 GB - the engine's other memory ~3 GB = ~7 GiB; 24 GB -> 14 GiB). The Mac's other
memory was still there: no memory pressure, macOS's GPU memory limit (about two thirds of RAM) not reached, and the
engine's actual footprint not measured. The engine has also grown since (MTP 0.45 GB, the staging copy 0.4 GB).

Other figures: with every expert in memory and no pauses, 29.6 tok/s; with 40 pauses per token, 26.3. Through Momoe's
server with Claude Code (`momoe run claude -p "What is 17*23? ..."`, answered 391): its 15,998-token prompt read in
68 s (236 tok/s), 9.8 tok/s writing at that length.

## Building and running

```sh
src/metal/build.sh            # patches and builds Momoe's copy of ggml and the engine into build-metal/
build-metal/momoe-metal -m models/ornith15-IQ4_XS/Ornith-1.5-35B-A3B-IQ4_XS.gguf --tokens 760,6511,314,9338,369 -n 64 \
    [--pool-gib 7] [--nocache] [--mtp 4 --draft-vocab data/draft_vocab_en.bin] [--ctx 4096]
momoe start ornith-mac        # config/ornith-mac.json: momoe-metal --serve, all experts, --ctx 65536, fit_max_tokens
momoe run claude
```

`src/metal/README.md` lists every option and environment switch.

## Flash-Next (qwen4exp)

Added 2026-10-12, measured on the same MacBook with Strata's IQ3_XXS file (3.83 GiB dense, 39.97 GiB of experts, 75 GB
in two files). The engine follows llama.cpp's qwen4exp graph: 4 hyper-connection streams (llama.cpp's fused Metal
ops), the n-gram embedding (PLE: 16 rows per token read on the CPU from the 28.8 GB table in the second file), the
delta net's sigmoid gate, and QSA, the sparse attention (each attention layer keeps an indexer key per position and a
pooled key per 4 positions; past 2,048 positions a token attends to its 512 best blocks and its own unfinished block;
llama.cpp's selection graph ported op for op, all on Metal). The file quantizes experts per layer (10 format
combinations): the pool has a group of slots per format.

Against llama.cpp on the CPU (it runs out of Metal memory with this file, also with the experts on the CPU):

- Two short prompts, 32 greedy tokens: identical, with 12, 6 and 3 GiB pools.
- Longer prompts: near ties flip (1,100 tokens: 0.66 / 0.33 here, 0.36 / 0.63 in llama.cpp; our own prompt chunk sizes
  move that step as much). At 8,015 tokens (QSA keeping a quarter of the blocks) both give the same top 5 tokens in the
  same order; without QSA the order differs. The top token's probability: 0.62-0.64 here, 0.42 in llama.cpp, not
  explained yet (llama.cpp's CPU matmuls round activations to 8 bits).

| | writing | reading a prompt |
|---|---|---|
| llama.cpp, CPU only | 4.2 tok/s | 8 tok/s |
| 12 GiB pool | 9.3-10.5 tok/s | 56-62 tok/s (1,100-8,015 tokens) |
| 6 GiB pool, SSD | 5.1 tok/s | |
| 3.4 GiB pool, SSD | 4.1 tok/s | |

Rough figures: 32-token answers, no `sudo purge` between runs. Through Momoe's server (`config/flashnext-mac.json`:
14 GiB pool, 32K context) `momoe run claude -p "What is 17*23? ..."` answered 391: Claude Code's 15,052-token prompt
in 303 s (49.6 tok/s), 6.2 tok/s writing.

Drafting (2026-10-13): the MTP layer comes from the checkpoint (`--mtp-gguf`, `tools/mtp_pack.py --experts q4_0`;
see src/metal/README.md). 12 GiB pool, code prompt, 200 tokens, 90 s pauses between runs: 6.8-7.3 tok/s without,
8.6-8.9 with `--mtp 3` (3.26 tokens per window), the same with `--mtp 2`. The same tokens as llama.cpp (CLI and
`--serve`).

## On a Mac Studio M1 Ultra (en), 2026-10-12

M1 Ultra (64-core GPU, 800 GB/s), 64 GB, macOS. llama.cpp b11429 (the same as on the MacBook), `llama-bench` and
the same `llama-server` runs as above; momoe-metal on the same prompts ("code", 15 tokens; second half of the answer).
Writing and prompt speed, tok/s:

| | Ornith IQ4_XS writes | Ornith reads | Flash-Next IQ3_XXS writes | Flash-Next reads |
|---|---:|---:|---:|---:|
| llama.cpp, Metal (`llama-bench`) | 96.6 | 1,647 (512), 1,573 (4K) | 34.9 | 566 (512) |
| llama.cpp, `llama-server`, 6 answers of 500 tokens | 88.9 | 1,507 (5.6K), 1,286 (20K) | 34.0 | 481 (5.6K), 465 (20K) |
| momoe-metal, every expert in memory, no pauses (`--preload`) | 54.6 | | 24.9 | |
| momoe-metal, every expert in memory, pauses (the default) | 24.7 | 582 (8K) | | |
| the same with drafting (`--mtp 4`) | 31.6 | | | |
| momoe-metal, 34 / 20 / 12 GiB pool | | | 12.3 / 12.3 / 12.4 | 214 / 207 (8K) |
| momoe-metal, 14 / 7 GiB pool, SSD | 23.7 / 21.6 | | | |

- llama.cpp runs Flash-Next on this 64 GB Mac (it failed on the 32 GB MacBook): the 70.6 GiB file is mapped, and what a
  token uses stays in memory (0.13 MB read from the SSD per token while writing).
- momoe-metal is no faster here than on the MacBook (Ornith 24.7 vs 25.5): the pauses dominate. Without them it is
  2.2x (Ornith) and 2.0x (Flash-Next) as fast. A pause costs ~550 us on the M1 Ultra (Ornith: 22 ms for 40 per
  token; Flash-Next ~850 us, 41 ms for 48), against ~110 us on the M4.
- Flash-Next's speed does not change between a 12 and a 34 GiB pool: at 12 GiB the misses already cost little next to
  the pauses.
- Even without pauses momoe-metal is behind llama.cpp (Ornith 54.6 vs 96.6, Flash-Next 24.9 vs 34.9).

Since then (2026-10-13), a pool that holds every expert runs without pauses (`--preload` made automatic). On the
MacBook, Ornith IQ4_XS, code prompt, 256 tokens: 28.4 tok/s (25.5 with pauses), 34.5 with drafting (31.6), against
llama.cpp's 33.7 (19.8 with its drafting); every expert read in 5.1 s at the start.

On en (2026-10-18, Flash-Next IQ3_XXS, a 27-token chat prompt, 256 tokens, second half): every expert in memory
(40.0 GiB, read in 7.2 s, no pauses) 26.7 tok/s, against llama.cpp's 34.0-34.9. With drafting too it does not fit:
Metal's working-set limit there is 51.5 GB, and the experts, the dense weights (3.83 GiB) and the MTP layer
(1.48 GiB) come to 45.3 GiB before the buffers ("Insufficient Memory"; at 16K context too). A pool of 34-37 GiB
with `--mtp 3` writes 21.6-21.8 (12.9-13.0 without): any expert missing keeps the pauses, so the pool's size above
34 GiB changes nothing; 38 GiB with drafting stopped with a bus error at the start.

What follows: on a Mac where llama.cpp can map the model, llama.cpp is faster. momoe-metal is useful only where it
cannot (Flash-Next on the 32 GB MacBook), and it would need pauses only on a miss (or none) to be worth more.

## llama.cpp against momoe-metal on both Macs (summary, 2026-10-13)

Writing speed in tok/s, gathered from the runs above and in src/metal/README.md (200-500-token answers, slightly
different prompts and days: differences under ~10% are noise). "Exact": the same tokens as llama.cpp; the
`--nostall` rows change the output.

MacBook Air M4 (zephyr), 32 GB, macOS 27.0:

| | Ornith IQ4_XS | Flash-Next IQ3_XXS |
|---|---:|---:|
| llama.cpp, GPU | **33.7** | fails: Metal out of memory |
| llama.cpp, with its MTP drafting | 19.8 | |
| llama.cpp, CPU only | | 4.2 |
| momoe, every expert in memory, exact | 28.4 | (does not fit) |
| momoe, every expert in memory, exact + MTP | **34.5** | |
| momoe, partial pool, exact (Ornith 7 GiB, Flash-Next 12 GiB) | 16.5 | 6.8-7.8 |
| momoe, partial pool, exact + MTP | | 8.6-8.9 |
| momoe, partial pool, `--nostall sub --nostall-every 8` | | 9.7 |
| momoe, partial pool, `--nostall sub` | 21.0 | 9.9 |
| reading a prompt, llama.cpp / momoe | 333 / | / 56-62 |

Mac Studio M1 Ultra (en), 64 GB, macOS 15.6.1:

| | Ornith IQ4_XS | Flash-Next IQ3_XXS |
|---|---:|---:|
| llama.cpp, GPU (`llama-bench` / `llama-server`) | **96.6 / 88.9** | **34.9 / 34.0** |
| momoe, every expert in memory, no pauses | 54.6 | 24.9 |
| momoe, partial pool, exact (Ornith 7 GiB, Flash-Next 34 GiB) | 22.8 | 11.8-12.3 |
| momoe, partial pool, exact + MTP | | 16.4 |
| momoe, partial pool, `--nostall sub --nostall-every 8` | 30.0 | 16.1 |
| momoe, partial pool, `--nostall sub` | 38.2 | 19.8 |
| momoe, partial pool, `--nostall sub` + MTP | | 31.7 (different text) |
| reading a prompt, llama.cpp / momoe | 1,647 / 582 (8K) | 566 / 214 (8K) |

- The MacBook is where momoe-metal is worth having: the only way to run Flash-Next there (2-2.4x llama.cpp on the
  CPU), and with drafting it matches llama.cpp on Ornith (34.5 vs 33.7).
- On the M1 Ultra the whole model fits and llama.cpp leads everywhere, also against momoe-metal without pauses.
- Not measured: Ornith with every expert in memory on en since preloading became automatic; prompt reading on the
  MacBook for a like-for-like pair.

## A 16 GB Mac, with the memory really missing (2026-10-18)

The MacBook (32 GB) with 16 GiB wired away by a small program (`mlock`: macOS can neither compress nor swap it, as
RAM a 16 GB Mac does not have), the owner's browsers left open, and setup's 16 GB config (Ornith IQ4_XS, a 5 GiB pool,
32K context, `--nocache`, `--mtp 4`) through the server. Greedy, three 400-token chats and one long prompt:

| | 16 GiB wired away | the same run, all 32 GB (the first try, when the wiring was released too early) |
|---|---:|---:|
| chat 1 / 2 / 3, writing | 11.2 / 9.7 / 13.1 tok/s | 11.3 / 9.3 / 13.8 tok/s |
| a 9,698-token prompt, reading | 126 tok/s | 114 tok/s |
| writing after it | 6.1 tok/s | 6.4 tok/s |

- The memory missing changes nothing measurable: the pool and the file-cache bypass already keep the engine to its
  own memory. Its footprint was 8.6 GB at the start and 9.35 GB at the end (`footprint`).
- macOS was at memory pressure "warn" (level 2) for 44 of the 45 samples (every 5 s), never "critical" once settled;
  it moved ~6 GB of the other programs to swap while the 16 GiB was wired.
- What this still does not show: a real 16 GB Mac's GPU limit. Metal's limit here stayed 26.8 GB
  (recommendedMaxWorkingSetSize: 78% of the 32 GB; the M1 Ultra's is 75% of 64 GB); on a 16 GB Mac it would be about
  11-12 GB if the share is the same, not far above the engine's 9.35 GB footprint. Only a real 16 GB Mac can say.

### llama.cpp the same way: it brings the Mac down

llama.cpp b11429 (`llama-bench -p 512 -n 128`) on the same Ornith IQ4_XS file (19.3 GB), with the same 16 GiB wired
away and ornith-mac stopped:

| layers on the GPU | what happened |
|---|---|
| all 40 (`-ngl 99`) | macOS hung about 2 minutes in, before the first token; kernel panic and restart |
| 20 of 40 (`-ngl 20`) | the same, about 2 minutes in |
| none (`-ngl 0`) | not tried |

- Both panics read "watchdog timeout: no checkins from watchdogd in 90 seconds". The system stopped running,
  rather than a program being stopped. At the panic llama-bench held about 120 MB but had read 1.17 million pages
  (16 KiB each, about the size of the file). In the second panic, 631 kernel threads were waiting on memory.
- The monitor (every 2 s, written to the disk) for the second run: the file was read at about 1.7 GB/s. About 25 s in,
  pressure went to "warn" and swap grew by about 1.2 GB every 2 s, to 2.9 GB at the last line. It never reached
  "critical", so the watchdog (critical for 5 s) never fired.
- Metal still offered 26.8 GB (recommendedMaxWorkingSetSize does not count wired memory), so llama.cpp sized itself
  for the whole Mac and not the 16 GB it really had.
- The browsers were holding about 6-7 GB in the first run and under 1 GB in the second, and both crashed the same way.
- momoe-metal ran under the same 16 GiB wire (above) because its pool and file-cache bypass keep it to about 9.35 GB.

## Next

1. Prompt speed: 236 tok/s on 16K tokens (llama.cpp 270-333); every chunk stages all 256 experts of each layer.
2. Writing at long context: 9.8 tok/s at 16K.
3. A 16 GB test on a real 16 GB Mac: Metal's GPU limit there (~11-12 GB?) against the engine's 9.35 GB footprint.
4. `momoe setup` on macOS: building momoe-metal, the config, the tokenizer folder without a Linux pack; setup's
   NVIDIA-specific messages.
5. Flash-Next: the 64 GB Mac Studio; Strata's MTP (mtp/rt); per-token speed (~100 ms with the experts in memory,
   ~35 ms of weight reads; the BF16 hyper-connection matrices are 1.3 GB per token).
6. Distribution: a ready-made binary from GitHub Releases once the repository is public: [mac-binary-releases.md](mac-binary-releases.md).
