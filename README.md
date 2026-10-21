<h1 align="center">MoMoE</h1>

<p align="center"><b>More MoEs: large mixture-of-experts models on one graphics card plus system RAM</b><br>
a fork of <a href="https://github.com/Niko1221/Strata">Strata</a> · tested on Linux with NVIDIA and on Apple Silicon Macs · free and open source (MIT)</p>

## Built on Strata

MoMoE is a fork of **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and the Strata contributors. Almost
everything that makes it work is theirs:
- the engine that runs a mixture-of-experts model with its experts spread over VRAM, RAM and the SSD;
- the expert cache and the predictions that fill it;
- the multi-token draft layer;
- the OpenAI- and Anthropic-compatible server;
- the installer;
- the measurements behind all of it.

How it works is written up in Strata's [paper](docs/paper/Strata-Paper.pdf) and [DETAILS.md](docs/DETAILS.md).

**If you want Qwen3.8-Flash-Next, use Strata.** It has a maintainer, ready-made engines for Windows and AMD cards, a
web app, and far more testing on far more hardware than this fork. Problems with Strata and improvements to it belong
in [Strata's issues](https://github.com/Niko1221/Strata/issues), not here.

## Why a fork

MoMoE started as a pull request to Strata ([#1402](https://github.com/Niko1221/Strata/pull/1402)) that added a second
model family. Some of what followed doesn't fit Strata, which takes one change per pull request and keeps its
default behaviour unchanged. MoMoE is the place for those changes:

- **More MoE models.** Strata runs Qwen3.8-Flash-Next and its variants. MoMoE also runs Qwen3.6-35B-A3B and its
  fine-tunes (Ornith 1.5, KAT-Coder), which fit 8-12 GB cards and 16-32 GB of RAM. The goal is more families: GLM and
  other large MoE models.
- **Agents first.** `momoe run claude` (or Codex, OpenCode, Unsloth Studio, LM Studio) starts the model and the tool,
  already connected to each other. There is no web page: the server is for your apps and agents.
- **Two platforms.** The aims are Linux with NVIDIA cards (CUDA) and Apple Silicon Macs, including Flash-Next on a
  32 GB MacBook by reading its experts from the SSD as they are needed. Macs have a first version (below). Other
  platforms and hardware (Windows, AMD or Intel cards) may work, but they will need help from people who run them.
- **Built from source.** MoMoE compiles its engine on your PC; there are no ready-made downloads.

## How it is written

Unlike my other projects, MoMoE's changes, since the fork and in the pull request it grew from
([#1402](https://github.com/Niko1221/Strata/pull/1402)), are written entirely with
[Claude Code](https://claude.com/claude-code). Strata's own code, which is most of what is here, is not covered by
this note.
— Aaron Flin ([aflin](https://github.com/aflin))

## Status

Early, and maintained by one person. MoMoE has three engines:
- **Strata's engine** (`momoe`), on Linux with an NVIDIA card: Qwen3.8-Flash-Next and the Qwen3.6 family.
- **MoMoE's ggml engine** (`momoe-cuda`), on Linux with an NVIDIA card: Nemotron 3.5 Bolt, GLM-5.3-Flash and
  LFM2.5 everywhere, and every family on a GPU that shares the system's memory (the DGX Spark, a Jetson), where Strata's
  engine is not built. On the Spark the experts that don't fit are read from the SSD directly, at 10-13 GB/s; on an
  8 GB Jetson they are copied from the file cache. Two cards share a model with the layer split or the helper tier.
- **MoMoE's Metal engine** (`momoe-metal`, [src/metal](src/metal/README.md)), on Apple Silicon Macs: every family
  above (as GGUF files). The experts that don't fit are read from the SSD as needed.

`momoe run` has been used with Claude Code, Codex, OpenCode, Unsloth Studio and LM Studio.

Not tested in MoMoE:
- Windows, and AMD or Intel cards;
- several GPUs with the Qwen3.6 family (it runs on one GPU);
- Swift 1.5 and Unsloth's Flash-Next files.

Strata's code for these is still here, but MoMoE has changed things around it.

## Models

The machines the models were tested on:

| Name | Machine |
| --- | --- |
| **PC** | Linux (Debian 12), NVIDIA RTX 4070 Ti (12 GB), Ryzen 9 5950X, 63 GB of RAM |
| **Spark** | NVIDIA DGX Spark: GB10, 128 GB of memory shared by the CPU and the GPU |
| **Jetson** | NVIDIA Jetson Orin Nano Super developer kit: 8 GB shared by the CPU and the GPU (7.3 GiB usable, about 5.9 GiB free beside its desktop), its 15 W power mode, an NVMe SSD |
| **M1 Ultra** | Mac Studio M1 Ultra, 64 GB |
| **M4 Air** | MacBook Air M4, 32 GB |
| **M4 Air as an 8 GB Mac** | the same MacBook with 10 GiB of its memory wired away (`mlock`): RAM an 8 GB Mac does not have |

### Which model runs where

Writing speed in tokens per second; "-" means not tried. Every model here was installed with `momoe setup` on the
machine; its greedy output was compared with llama.cpp's (48 or 64 tokens: the same, or parting only where the top two
tokens are within a few hundredths of each other, which was checked); and the server answered a chat, made a tool call
(whole and streamed) and answered after the tool's result.

| Model, size | PC | Spark | Jetson | M1 Ultra | M4 Air |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next IQ3_XXS | 57.9* | 35.5 | - | 38.2 | 7.9* |
| Flash-Next Coder IQ1_M | 54.8* | 33.3 | - | 48.9* | - |
| Swift 1.5 IQ3_XXS | 63.4* | 34.6 | - | 37.9 | - |
| Flash-Next (Unsloth) UD-IQ4_XS | 12.4* | 30.2 | - | 19.0* | - |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 158.7* | 125.2* | - | 129.3* | - |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 157.5* | 123.6* | - | 117.1* | - |
| Ornith-1.5-35B-A3B IQ4_XS | 128.8* | 104.1* | - | 92.7* | 31.2* (as a 16 GB Mac, earlier: 9.7-13.1) |
| KAT-Coder-V2.5-Dev i-quality-v2 | 126.2* | 99.4* | - | 104.9* | - |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 29.7 | 28.4 | - | 37.5 (UD-Q3_K_XL) | - |
| Ornith-1.5-122B-A10B UD-IQ3_XXS | 19.2 | 33.2* | - | 19.8* | 4.5 |
| GLM-5.3-Flash UD-IQ3_XXS | does not fit a 12 GB card | 19.7* | - | 4.5* | - |
| LFM2.5-24B-A2B | 75.9 (UD-Q4_K_XL) | 134.0 | 6.9 | 104.3 | 11.9 (as an 8 GB Mac) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 305.5 | 161.7 | 16.6 | 185.3 | 43.1 (as an 8 GB Mac) |
| Flash-Next NVFP4 | Strata's own packs: a PC with an NVIDIA card only, not tested in MoMoE | | | | |

\* With drafting: the model's own multi-token-prediction (MTP) layer proposes up to 3 or 4 tokens ahead and the model
checks them in one pass, keeping only what it would have written itself. Numbers without \* are without drafting:
the Flash-Next family on the Spark, and Flash-Next IQ3_XXS and Swift on the M1 Ultra (where every expert fits only
without the draft layer, which measured faster) were measured without it, LFM has no MTP layer, and MoMoE does not yet draft with Bolt's.
Ornith-122B drafts only where (nearly) every expert is in memory, and setup adds `--mtp 3` only there (the Spark 26.4
-> 33.2 tok/s, the M1 Ultra 16.4 -> 19.8): on the PC the CPU computes the experts the card does not hold and drafting
was slower (19.1 -> 12.5), on the M4 Air (an 11 GiB pool, the rest from the SSD) 4.1-4.5 -> 4.0. Its greedy tokens
against llama.cpp's (64): the same on both Macs and on the Spark with drafting; `momoe-cuda` without drafting parts at
token 33, a three-way choice where each engine puts the weights differently (llama.cpp 0.53 / 0.28 / 0.19, `momoe-metal`
0.56 / 0.39 / 0.05, `momoe-cuda` 0.36 / 0.38 / 0.25). Over 300 tokens fed llama.cpp's own, both are as close to it
(KL from llama.cpp's top 20, mean / median: `momoe-cuda` on the Spark 0.0107 / 0.0006, `momoe-metal` on the M4 Air
0.0108 / 0.0006; the same top token 98.3% and 99.0%; Ornith-35B: 0.0028 and 0.0030). llama.cpp's numbers: with drafting where marked \* (see below), else without.

On the Macs, `momoe-metal` keeps one thread waiting on the CPU's wait-for-event instruction while a request runs.
Without it, macOS held the engine's main thread on the efficiency cores and the GPU's clock low, at random from run to
run and on every request through the server (LFM2.5-8B on the M1 Ultra, through `momoe start`: 64-69 tok/s; with it
120-128). Thread priorities and launchd's `Interactive` did not change it.

`momoe-metal` also keeps its graphs, as llama.cpp does: a graph is built once for its shapes (the tokens, the
attention window, the outputs read back, the k-pool blocks a batch finishes) and computed again with the next token's
inputs, the KV caches written at rows given as an input. Building it each token kept the GPU waiting (LFM2.5-8B: 0.83
ms of 8.3), and the GPU's clock follows how soon the next token's work arrives: llama.cpp with one thread, or without
polling, wrote 78 tok/s at ~800 MHz; with two polling threads 182 at ~1,290 MHz. With graphs kept, LFM2.5-8B went from
120 to 185 tok/s (its server: 177.5) and LFM2.5-24B from 57 to 104; the other models 7-20% faster; the same tokens as
before on every model, with graphs kept or not. Only where every expert is in memory: with an expert pool (Unsloth's
Flash-Next, GLM) a graph is built each token, as before. `MOMOE_NO_GRAPH_REUSE=1` turns it off.

Two more on `momoe-metal`, from comparing its graph with llama.cpp's node by node (Qwen3.6 without drafting, 76.5 ->
92.2 tok/s, llama.cpp 90.3): the delta net's new state is written into its cache as one row, the shape ggml-metal's
GATED_DELTA_NET + CPY fusion takes, so the kernel writes it in place (it was a 2 MB copy after each of the 30 layers:
-1.8 ms a token); and with every expert in memory, the router's choice goes to the experts as it is (no copy of it, no
slot lookup: 80 small kernels a token fewer). The same tokens as before on every model but the Coder, which parts at
token 263: with 10 experts a token (no fused reduce for more than 8) ggml-metal adds them in another order, and on this
prompt fewer of its drafts were kept (48.9 tok/s, before 56.0; on two other prompts faster: 40.8 -> 46.8, 36.8 ->
37.4). `MOMOE_NO_DIRECT=1` turns the second off. On `momoe-cuda` only the first (no faster there: CUDA graphs).

`momoe-cuda` keeps its graphs the same way, and a kept graph runs as a CUDA graph (its kernels launched together, not
one by one; graphs kept without it: 283): LFM2.5-8B on the PC 255 -> 305.5 tok/s (llama.cpp 305.8), on the Jetson with every expert in memory 18.2 ->
20.3 (llama.cpp 20.5), on the Spark 5-9% faster. Not for Bolt with its experts on the CPU (its own kept decode graph)
nor across two cards (not measured). Drafting on `momoe-cuda` also copied the whole token-embedding table from RAM to
the GPU for every drafted token (540 MB for Qwen3.6, ~10 ms on the Spark); its rows are now looked up on the CPU, as
the model's own pass did: on the Spark, Qwen3.6 59.5 -> 125.2 tok/s, KAT-Coder 57.5 -> 99.4, Ornith 66.2 -> 104.1.
The same tokens as before on every model. GLM-5.3 drafts without that copy: 19.4 -> 19.7.

Which engine: on the PC, Strata's engine for the Qwen3.8-Flash-Next and Qwen3.6 families (Ornith with MoMoE's ggml
engine: 63.9) and MoMoE's ggml engine (`momoe-cuda`) for Bolt, GLM and LFM; `momoe-cuda` for everything on the Spark
and the Jetson; `momoe-metal` on the Macs.

### Against llama.cpp on the same machine

| Model | Machine | MoMoE | llama.cpp |
| --- | --- | ---: | ---: |
| LFM2.5-24B-A2B UD-IQ3_XXS | Jetson | 6.9 (engine) / 7.0 (server) | 4.8 (16 layers' experts on the GPU, the most that fit) / 3.7-4.3 (server) |
| LFM2.5-8B-A1B UD-Q4_K_XL | Jetson | 16.6 | 17.2 (18 of 24 layers' experts on the GPU) |
| LFM2.5-8B-A1B UD-IQ3_XXS | Jetson | 20.3 (every expert in memory, `--pool-gib 2.7`) | 20.5 (all on the GPU) |
| LFM2.5-24B-A2B UD-Q4_K_XL | PC | 75.9 | 105.1 (10 layers' experts on the CPU, the best fit) |
| LFM2.5-8B-A1B UD-Q4_K_XL | PC | 305.5 | 305.8 (all on the card) |
| LFM2.5-8B-A1B UD-Q4_K_XL | M1 Ultra | 185.3 / 177.5 (server) | 169-190 / 176-182 (server) |
| LFM2.5-24B-A2B UD-IQ3_XXS | M1 Ultra | 104.3 / 99.8 (server) | 108.7 |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | M1 Ultra | 129.3* / 92.0 | 150.1* (4 drafted) / 90.3 |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | M1 Ultra | 117.1* / 95.8 | 146.1* (4) / 96.1 |
| KAT-Coder-V2.5-Dev i-quality-v2 | M1 Ultra | 104.9* / 91.0 | 106.8* (3) / 89.6 |
| Ornith-1.5-35B-A3B IQ4_XS | M1 Ultra | 92.7* / 96.1 | 96.2* (2) / 96.1 |
| Nemotron 3.5 Bolt UD-Q3_K_XL | M1 Ultra | 37.5 | 38.5 (it cannot read a prompt: its Metal build has no kernel for 9 experts a token) |
| Nemotron 3.5 Bolt UD-IQ3_XXS | PC | 30.0 / 30.6 (server, greedy / sampled) | 26.2 / 26.1 |
| Ornith-1.5-122B-A10B UD-IQ3_XXS | PC | 19.2 (the CPU computes the experts the card does not hold) | 4.62 (48 layers' experts on the CPU, the best fit) |
| GLM-5.3-Flash UD-IQ3_XXS | Spark | 19.7* / 15.0* / 14.6* (server, greedy / sampled); 13.3 | 1.3 (a 48-token check: every expert on the CPU, cold, not a speed test) |
| GLM-5.3-Flash UD-IQ4_XS | Spark | 11.7* / 10.4* (server); 9.3 | 2.3 / 2.5 |

\* With drafting (see above); the others without. llama.cpp drafts with the same MTP layer (`llama-server
--spec-type draft-mtp`, b11724), shown at its best number of drafted tokens (2-4); MoMoE drafts 4. Both sides for these
rows: greedy, the same prompt, 400 tokens (MoMoE: its engine, the answer's second half; llama.cpp: its server's own
timing over all 400). Without drafting the two are level (92.0-96.1 tok/s against 89.6-96.1); llama.cpp gains more
from drafting (Qwen3.6: 1.66x against 1.41x), the gap left. Ornith's MTP layer is a weak drafter in both (llama.cpp: 54%
of its drafts kept at 2, 28% at 4): without drafting both write 96.1. On the M4 Air, where it runs from a 14 GiB
pool and the SSD, drafting still wins (30.7-31.7 against 24.6-25.2 tok/s without: a window shares the experts' reads).

Where every expert fits on the GPU, MoMoE with its graphs kept is level with llama.cpp on LFM2.5 (the PC, the Jetson,
the M1 Ultra), and on the Qwen3.6 family on the M1 Ultra without drafting; with drafting llama.cpp is ahead there. MoMoE gains where the experts do not fit: the Jetson,
GLM on the Spark, Bolt on the PC.

Reading prompts on the PC, with Strata's engine (4K / 16K tokens): Flash-Next IQ3_XXS 1,178 / 2,133 tok/s; the Coder
1,931 / 2,615; Swift 1,196 / 2,163; Unsloth's UD-IQ4_XS 67 / 91; the Qwen3.6 family 3,081-3,767 / 3,974-4,521.

How these were measured (2026-10-19 unless named; the M1 Ultra and Spark columns and LFM2.5-8B on the PC 2026-10-20; build `momoe-cuda` / `momoe-metal` with ggml from llama.cpp b11724):
- **Strata's engine on the PC:** `tools/strata_bench.py`, three runs, medians: "chat decode" in the table.
- **`momoe-cuda` and `momoe-metal`:** the engine on its own, greedy, the second half of a 300-400-token answer, with
  the arguments setup wrote (drafting where they have it); "server" rows: the server's own speed over a long chat.
- **The Spark:** one model at a time under a watchdog. GLM-5.3-Flash with `--mtp 3`, 160 tokens; its server and the runs without drafting, 2026-10-17. UD-IQ3_XXS (113 GiB, setup's choice on the Spark) is a quarter smaller than UD-IQ4_XS (146
  GiB): nearly all its experts fit in memory (97.6% of expert reads from the pool) and it writes 1.3-1.4 times as
  fast, more with drafting.
- **The Jetson:** 16K context. llama.cpp b11724, `llama-bench` tg128.
- **The Macs:** setup's configs: every expert in memory where it fits beside the KV cache (Flash-Next IQ3_XXS and Swift
  at 32K context, the others at 64K), else an expert pool and the rest from the SSD (Unsloth's Flash-Next: 36 GiB on the
  M1 Ultra; GLM: 29 GiB; on the M4 Air, Flash-Next IQ3_XXS 11 GiB and Ornith 14 GiB, from the memory free when setup
  ran, with Firefox open). The Flash-Next family drafts with Strata's draft layer (its ~5 GB of MTP tensors from Qwen's
  checkpoint, packed Q2_0, as Strata's setup does on a PC) unless every expert fits only without it (Flash-Next IQ3_XXS
  on the M1 Ultra, before graphs were kept: 29.7 tok/s without it, 24.3 drafting from a 37 GiB pool; Unsloth's
  UD-IQ4_XS: 19.0 drafting from 36 GiB, 12.4 without from 39). Unsloth's Flash-Next and GLM on the M1 Ultra: 2026-10-19
  (an expert pool: no graphs kept). The files larger than Metal's working set (48 GiB on the M1 Ultra) were not run in
  llama.cpp: that took a MacBook down. The 8 GB-Mac rows: setup's 8 GB config (a 3 GiB pool, 16K context) with 10 GiB
  wired away; with all 32 GB the same runs wrote 43.3 and 12.0 tok/s (14.8 for the 24B before; the same with the
  wait-for-event thread off). It does not simulate an 8 GB Mac's smaller Metal limit.
- **Before today:** Ornith on the M4 Air, its 16 GB simulation and llama.cpp beside them are in [design/mac-port.md](design/mac-port.md).

### Two cards: layer split and helper tier

`momoe-cuda` takes the same switches as Strata's engine: a config with `"gpu": [a, b]` splits the layers across the
cards (`"layer_split": "auto"`, or the first layer of each later card), and `--helper-tier` in its `"args"` makes the
second card a second tier of experts under the first one's (see [the helper tier](#two-different-gpus-the-helper-tier)).
Setup writes them with `--gpus a,b` (and `--helper-tier`), and splits GLM-5.3-Flash by itself across two cards under
16 GB each. Not yet with the CPU's experts (Bolt's `--hybrid`) or the mailboxes; not yet measured for this table.

`./setup.sh --check` lists what fits your PC. The Qwen3.6-family files are described in
[docs/QWEN36.md](docs/QWEN36.md) and [docs/QWEN35MOE_GGUFS.md](docs/QWEN35MOE_GGUFS.md). The authors and quantizers
above made these models; MoMoE only runs them.

## Install

You need:
- an NVIDIA card, RTX 20 series or newer;
- enough RAM for the model;
- Python 3;
- a C++ compiler and the CUDA toolkit. Setup checks your PC and offers to install them if they are missing.

On an Apple Silicon Mac (16 GB or more) you need Xcode's command line tools (`xcode-select --install`) and Python
3.10 or newer (`brew install python`); setup builds the Metal engine and downloads Ornith-1.5 (19.3 GB).

```sh
git clone https://github.com/aflin/momoe.git
cd momoe
./momoe setup                         # check the PC, choose a model, compile the engine, download, start
sudo ln -s "$PWD/momoe" /usr/local/bin/momoe    # optional: run momoe from any folder
```

Setup asks for a name for the model's config (for example `ornith-8gb`) and starts the server in the background.

## Use

```sh
momoe list                            # the configs in config/
momoe start ornith-8gb                # start one (.json can be left off)
momoe status                          # what is running, and where
momoe log                             # follow the server's and the engine's logs (Ctrl+C ends)
momoe run claude                      # Claude Code, connected to it (also: codex, opencode)
momoe run unsloth                     # Unsloth Studio with a "MoMoE" provider
momoe run lmstudio                    # LM Studio, with the plugin it needs installed and set up
momoe run claude --config ornith-8gb  # start that config first if nothing is running
momoe stop
```

When no server is running, `momoe run` starts one, asking which config if there are several. It gives the tool its
settings for that run only and installs nothing: a missing tool is named, with how to install it.

Any OpenAI- or Anthropic-compatible client can use the server directly at `http://127.0.0.1:<port>/v1`; the port is
in the config, and `momoe status` shows it. The server listens on 127.0.0.1 only. Reaching it from another device needs
`--host` together with `--api-key`, for example `momoe start ornith-8gb --host 0.0.0.0 --api-key KEY`. Given to
`momoe start`, they apply to that start only; add `--save` to keep them in the config.

### Logs

Every log is in `logs/`. Each config has its engine's log (`logs/<config>.log`: loading, the expert cache, one block of
numbers per request). The server writes `logs/server.log`: a line a second while it answers, with its speed, and a
`done:` line per answer with the expert cache's hit rate. `logs/server.log` is started again with each `momoe start`.
The engine logs keep growing.

```sh
momoe log                             # the last 20 lines of the server's and the engine's logs, then follow them
momoe log server                      # one log: server, engine, unsloth or lmstudio (all: the four)
momoe log -n 100                      # start from the last 100 lines
momoe log --no-follow                 # print the last lines and stop
```

With two logs or more, each line starts with `[server]`, `[engine]`, ... The engine log is the running config's
(when none runs, the last one started). Ctrl+C stops following; the server keeps running.

## Two different GPUs: the helper tier

With two different NVIDIA cards, for example a fast one and an older one, `--helper-tier` uses the second card as a
second level of expert cache instead of splitting the model's layers between the cards:

1. the main card runs the model as it would alone, with its own expert cache;
2. the helper card holds the next experts in line and computes them itself;
3. the RAM holds the rest.

Each expert is in one place only. When the main card takes in an expert from the helper, the one it lets go moves
down to the helper; the helper's evictions move down to RAM. The experts on the helper need no RAM, which is the point:
**the PC keeps much more memory free**, at about the same speed.

Setup offers it when it finds two different cards that can run the model together (the layer split stays its
recommendation), or `./momoe setup --helper-tier`. By hand: both cards in the config's `"gpu"`, the main card first,
and `--helper-tier` in `"args"`:

```json
"gpu": [1, 0],
"args": [..., "--helper-tier"]
```

`--helper-gib G` limits the helper's VRAM for experts (default: what fits, less 1.6 GB kept free for its prompt
buffers). Every setting it turns on can be changed on its own: see
[docs/SECOND_GPU.md](docs/SECOND_GPU.md#the-helper-tier---helper-tier).

Measured on the PC, RTX 4070 Ti (main) + RTX 3060 12 GB (helper, also driving the desktop), Flash-Next IQ3_XXS, 64K
context, `tools/strata_bench.py`, 3 runs per test, 2026-10-18:

| | 4070 Ti alone (two runs) | `--helper-tier` |
| --- | ---: | ---: |
| experts on the 3060 | | 5,492 (8.83 GiB) |
| experts in RAM | ~41 GiB | 29.9 GiB |
| the PC's available memory | 2.1 GB | 11.8 GB |
| writes, short chat | 58.2 / 55.9 | 59.7 |
| writes, agent session | 63.7 / 63.8 | 68.3 |
| reads, 4K-token prompt | 1,178 | 1,478 |
| reads, 16K-token prompt | 2,130 | 1,813 |
| agent turn, first token | 3.37 s | 2.51 s |

Writing speed is about the same: the agent session moved between 64.5 and 71.1 between runs of the same
configuration. Short prompts are read faster and long ones slower. The answers are not bit-identical to one card's.

With a 262K context, the KV cache leaves the 4070 Ti room for fewer experts (1,071 instead of about 2,850). In
everyday use with that config (2026-10-18, 7 answers of 115-4,572 tokens), it wrote 43.6-56.1 tok/s with a 54-71%
hit rate in the main card's cache. The first answer after each start was about 20 tok/s, while the caches fill.

For two equal cards, the layer split should be the better choice: each card runs its own layers without waiting for
the other inside a layer. That, other cards, other models and Windows are not measured.

## Where things are

| Folder | What |
| --- | --- |
| `config/` | the model configs setup writes (yours; not in git) |
| `models/`, `packs/`, `mtp/` | the downloaded models and what setup prepares from them (not in git) |
| `engine/` | the compiled engine (`momoe`, `momoe-vision`) |
| `logs/` | every log: the engine's (`<config>.log`), the server's (`server.log`: its speed and cache hit lines), Unsloth Studio's and LM Studio's |
| `.momoe/` | the server's state (which config runs) |
| `design/` | MoMoE's design notes: what was measured, chosen or rejected |
| `docs/` | Strata's documentation. Most of it applies; it says "Strata" where MoMoE has its own name |

## Credits

- **[Strata](https://github.com/Niko1221/Strata)**, by Niko1221 and the Strata contributors: the engine, the server,
  the installer, the documentation and the paper. MoMoE is their work with changes on top.
- Changes taken from other Strata forks and open pull requests, credited to their authors:
  - [Asofwar/Strata](https://github.com/Asofwar/Strata) by Asofwar: the Qwen3.6 decode-path speedups, the fast
    256 x 8 router on CUDA and the control-vector fix (three commits, under their own name).
  - [#2674](https://github.com/Niko1221/Strata/pull/2674) by Scott Page ([page1](https://github.com/page1)): the
    server logs a reply that ended after its thinking with no text and no tool call.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**: the GGUF format, the quantization kernels and the
  image encoder, which Strata builds on.
- The models' authors and quantizers listed above: the Qwen team, ISTA-DASLab, ukisai, Unsloth, bartowski, Ornith AI
  and Myric.

## License

MIT, like Strata: see [LICENSE](LICENSE). Strata's copyright notice stays with its code, and MoMoE's changes are under
the same license.
