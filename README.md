<h1 align="center">MoMoE</h1>

<p align="center"><b>More MoEs: large mixture-of-experts models on one graphics card plus system RAM</b><br>
its own CUDA and Metal engines, and <a href="https://github.com/Niko1221/Strata">Strata</a>'s · tested on Linux with NVIDIA and on Apple Silicon Macs · free and open source (MIT)</p>

## Quick start

You need Linux with an NVIDIA card (RTX 20 series or newer) or an Apple Silicon Mac, Python 3, and enough RAM for
the model. Setup checks the rest and says what is missing.

```sh
git clone https://github.com/aflin/momoe.git
cd momoe
./momoe setup                  # pick a model: setup compiles the engine, downloads the model and starts it
```

Setup picks the model's size for your machine and writes a config for it (for example `flint-ud-iq3_xxs`); to skip
the menu, name the family: `./momoe setup --family flint`. Then:

```sh
./momoe run claude             # Claude Code connected to the model (also: codex, opencode, unsloth, lmstudio)
./momoe status                 # what is running, and its address (http://127.0.0.1:8080/v1 by default)
./momoe stop
./momoe start flint-ud-iq3_xxs # start a config again later (./momoe list shows them)
./update.sh                    # the newest MoMoE; compiles again only what changed
```

Any OpenAI- or Anthropic-compatible app can use the address `momoe status` shows. More in [Install](#install),
[Use](#use) and [Models](#models).

## What MoMoE is

MoMoE began as a fork of **[Strata](https://github.com/Niko1221/Strata)**, by Niko1221 and the Strata contributors,
and Strata is still its base. It has three engines:

| Engine | Runs on | Models |
| --- | --- | --- |
| **Strata's engine** (`momoe`) | Linux + NVIDIA | Qwen3.8-Flash-Next and its variants, the Qwen3.6 family |
| **`momoe-cuda`** (MoMoE's own) | Linux + NVIDIA, the DGX Spark, a Jetson | every family below; on a PC: Nemotron 3.5 Bolt and Lightning, GLM-5.3-Flash, Ornith-1.5-122B, Flint, Gemma 4, LFM2.5 |
| **`momoe-metal`** (MoMoE's own) | Apple Silicon Macs | every family below |

- **Strata's engine** is Strata's, with MoMoE's additions (the Qwen3.6 family's path began as
  [#1402](https://github.com/Niko1221/Strata/pull/1402)): experts spread over VRAM, RAM and the SSD, an expert cache
  filled by predictions, and a multi-token draft layer. How it works: Strata's [paper](docs/paper/Strata-Paper.pdf) and
  [DETAILS.md](docs/DETAILS.md). The server and the installer grew from Strata's too.
- **MoMoE's engines** ([src/metal](src/metal/README.md)) are written for MoMoE on
  [ggml](https://github.com/ggml-org/llama.cpp), llama.cpp's library. They read a model's GGUF as it is and keep the
  experts they can in a pool on the GPU. The rest are, per model and machine:
  - read from the SSD (on the Spark past the page cache, at 10-13 GB/s);
  - copied in from RAM;
  - or computed by the CPU while the GPU computes its own (for Ornith-122B and Flint, a trial while the model runs
    picks the faster way).

  Their graphs are kept and replayed (as CUDA graphs on NVIDIA), and two cards can share a model (the layer split or
  the [helper tier](#two-cards)).

**For Qwen3.8-Flash-Next on Windows, an AMD or Intel card, or with a desktop app, use Strata.** It has a maintainer,
ready-made engines and far more testing on far more hardware. Problems with Strata's engine that also happen in
Strata belong in [Strata's issues](https://github.com/Niko1221/Strata/issues).

Why a separate project: MoMoE started as a pull request to Strata ([#1402](https://github.com/Niko1221/Strata/pull/1402))
that added a second model family, and some of what followed doesn't fit Strata, which takes one change per pull request
and keeps its default behaviour unchanged. MoMoE's aims:
- **more MoE models**, each installed with `momoe setup`;
- **agents first:** `momoe run claude` (or Codex, OpenCode, Unsloth Studio, LM Studio) starts the model and the tool,
  already connected; there is no web page;
- **Linux with NVIDIA, and Apple Silicon Macs** (Windows and AMD or Intel cards may work, but need help from people
  who run them; with the Qwen3.6 family MoMoE runs on one GPU);
- **built from source:** setup compiles the engine on your machine.

Status: early, and maintained by one person.

### How it is written

Unlike my other projects, MoMoE's changes, since the fork and in the pull request it grew from
([#1402](https://github.com/Niko1221/Strata/pull/1402)), are written entirely with
[Claude Code](https://claude.com/claude-code). Strata's own code, which is most of what is here, is not covered by
this note.

## Models

The machines:

| Name | Machine |
| --- | --- |
| **PC** | Debian 12, RTX 4070 Ti (12 GB), Ryzen 9 5950X, 63 GB of RAM |
| **Spark** | NVIDIA DGX Spark, 128 GB shared by CPU and GPU |
| **Jetson** | Jetson Orin Nano Super, 8 GB shared (about 5.9 GiB free beside its desktop), 15 W mode, NVMe SSD |
| **M1 Ultra** | Mac Studio M1 Ultra, 64 GB |
| **M4 Air** | MacBook Air M4, 32 GB ("as an 8 GB Mac": setup's 8 GB config with 10 GiB locked away) |
| **Mac mini** | Mac mini M1, 8 GB, macOS 11 |

### Which model runs where

Writing speed in tokens per second, with setup's config on each machine; "-" means not tried. \* = drafting with the
model's own multi-token-prediction layer. These are short tests: a short prompt (a few short questions, or a
few hundred tokens) on setup's config, whose context is small (16K to 64K tokens). Real use is slower, for two
reasons. A larger context leaves less memory for the expert cache: Ornith-1.5-35B on the PC wrote 128.8
tok/s with setup's 32K config (a 7.6 GiB cache) and 99.0 with a 256K one (5.1 GiB). And the attention over a long
conversation costs more for each new token: in a Claude Code session on that 256K config it wrote about 85 tok/s at 40K
tokens of conversation and 55-60 at 100K, with the same share of experts found in the cache (about 75-80%) at both.

| Model, size | PC | Spark | Jetson | M1 Ultra | M4 Air |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next IQ3_XXS | 57.9* | 35.5 | - | 38.2 | 7.9* |
| Flash-Next Coder IQ1_M | 54.8* | 33.3 | - | 48.9* | 12.4* |
| Swift 1.5 IQ3_XXS | 63.4* | 34.6 | - | 37.9 | 11.4* (IQ2_XS) |
| Flash-Next (Unsloth) UD-IQ4_XS | 12.4* | 30.2 | - | 19.0* | 6.1* |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 158.7* | 125.2* | - | 129.3* | 56.5* |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 157.5* | 123.6* | - | 117.1* | 60.2* |
| Ornith-1.5-35B-A3B IQ4_XS | 128.8* | 104.1* | - | 92.7* | 49.3* (a 14 GiB pool: 31.2*) |
| KAT-Coder-V2.5-Dev i-quality-v2 | 126.2* | 99.4* | - | 104.9* | 41.1* |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 29.7 | 28.4 | - | 37.5 (UD-Q3_K_XL) | 8.6 |
| Ornith-1.5-122B-A10B UD-IQ3_XXS | 19.2 | 33.2* | - | 19.8* | 4.5 |
| GLM-5.3-Flash UD-IQ3_XXS | does not fit a 12 GB card | 19.7* | - | 4.5* | 1.4 |
| Flint 40B-A2.5B | 89.9 (UD-IQ3_XXS) / 65.7 (UD-Q4_K_XL) | 73.2 (UD-Q4_K_XL) | - | 84.4 (UD-Q4_K_XL) | 42.2 (UD-IQ3_XXS) |
| Nemotron 3.5 Lightning 30B-A3B | 71.1 (UD-IQ3_XXS) / 38.7 (UD-Q4_K_XL) | 66.9 (UD-Q4_K_XL) | - | 89.9 (UD-Q4_K_XL) | 25.9 (UD-IQ3_XXS) |
| Gemma 4 26B-A4B | 92.4 (UD-IQ3_XXS) / 75.2 (UD-Q4_K_XL) | 59.3 (UD-Q4_K_XL) | - | 85.2 (UD-Q4_K_XL) | 28.8 (UD-IQ3_XXS) |
| LFM2.5-24B-A2B | 75.9 (UD-Q4_K_XL) | 134.0 | 6.9 | 104.3 | 59.0 (11.9 as an 8 GB Mac) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 305.5 | 161.7 | 16.6 | 185.3 | 43.1 (as an 8 GB Mac) |
| Flash-Next NVFP4 | Strata's own packs: a PC with an NVIDIA card only, not tested in MoMoE | | | | |

- **Engines:** on the PC, Strata's engine runs the Flash-Next and Qwen3.6 families, and `momoe-cuda` runs the rest
  (Ornith-35B on `momoe-cuda` would be 63.9). Elsewhere it's `momoe-cuda` on the Spark and the Jetson, and
  `momoe-metal` on the Macs.
- **Checked:** every model's greedy output against llama.cpp's, and the server's chat and tool calls (whole, streamed,
  after the result).
- **Reading prompts on the PC** (Strata's engine, 4K / 16K tokens): Flash-Next IQ3_XXS 1,178 / 2,133 tok/s, the
  Coder 1,931 / 2,615, Swift 1,196 / 2,163, Unsloth's UD-IQ4_XS 67 / 91, the Qwen3.6 family 3,081-3,767 /
  3,974-4,521.
- **Details:** how each column was measured, drafting per model, and the engines' speed-ups:
  [design/benchmark-notes.md](design/benchmark-notes.md). The raw numbers are in [bench/results](bench/results).

### Against llama.cpp on the same machine

Writing speed in tokens per second. llama.cpp is shown at the best of the placements tried: "N on CPU" is
`--n-cpu-moe N` (the experts of N layers in RAM); "MoE cache G" is `--cpu-moe --moe-cache-mib` (every expert in RAM, a
cache of G on the GPU, llama.cpp's own expert cache since 2026-10-07). \* = drafting, both sides with the same MTP
layer. "MoMoE" is with setup's config.

| Model | MoMoE | llama.cpp |
| --- | ---: | ---: |
| **PC** (RTX 4070 Ti 12 GB) | | |
| Qwen3.8-Flash-Next IQ3_XXS | 65.5-76.7* | 23.1 (MoE cache 5 GiB; its default fit 22.5) |
| Ornith-1.5-35B-A3B IQ4_XS | 131.6-153.5* | 77.8 (MoE cache 8 GiB; its default fit 57.2; drafting 63.5*) |
| Ornith-1.5-122B-A10B UD-IQ3_XXS | 19.2 | 4.62 (48 on CPU) |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 30.0 (server, sampled 30.6) | 26.2 (26.1) |
| Nemotron 3.5 Lightning UD-IQ3_XXS / UD-Q4_K_XL | 71.1 / 38.7 | 65.3 (24 on CPU) / 40.2 (36) |
| Gemma 4 26B-A4B UD-IQ3_XXS / UD-Q4_K_XL | 92.4 / 75.2 | 90.5 at 32K context (4 on CPU) / 54.8 (11) |
| Flint UD-IQ3_XXS / UD-Q4_K_XL | 89.9 / 65.7 | 102.6 (15 on CPU) / 58.5 (30) |
| LFM2.5-24B-A2B UD-Q4_K_XL | 75.9 | 105.1 (10 on CPU) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 305.5 | 305.8 |
| **Spark** (128 GB shared) | | |
| GLM-5.3-Flash UD-IQ3_XXS | 19.7* (13.3 without) | 10.85 (5 on CPU; `--no-host 1 --repack 0 -lm mmap`) |
| GLM-5.3-Flash UD-IQ4_XS | 11.7* (9.3 without) | 3.52 (20 on CPU, the same settings; larger than the memory) |
| Flint / Gemma 4 / Lightning UD-Q4_K_XL | 73.2 / 59.3 / 66.9 | 76.0 / 57.5 / 69.6 |
| **Jetson** (8 GB shared) | | |
| LFM2.5-24B-A2B UD-IQ3_XXS | 6.9 | 4.8 (16 layers' experts on the GPU, the most that fit) |
| LFM2.5-8B-A1B UD-Q4_K_XL / UD-IQ3_XXS | 16.6 / 20.3 | 17.2 / 20.5 |
| **M1 Ultra** (64 GB) | | |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 129.3* / 92.0 | 150.1* / 90.3 |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 117.1* / 95.8 | 146.1* / 96.1 |
| KAT-Coder-V2.5-Dev i-quality-v2 | 104.9* / 91.0 | 106.8* / 89.6 |
| Ornith-1.5-35B-A3B IQ4_XS | 92.7* / 96.1 | 96.2* / 96.1 |
| Nemotron 3.5 Bolt UD-Q3_K_XL | 37.5 | 38.5 (cannot read a prompt: no Metal kernel for 9 experts a token) |
| Flint / Gemma 4 / Lightning UD-Q4_K_XL | 84.4 / 85.2 / 89.9 | 77.8 / 86.7 / 92.1 |
| LFM2.5-24B-A2B UD-IQ3_XXS / LFM2.5-8B-A1B UD-Q4_K_XL | 104.3 / 185.3 | 108.7 / 169-190 |
| **M4 Air** (32 GB) | | |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 60.8* / 33.2 | 47.6* / 28.9 |
| Ornith-1.5-35B-A3B IQ4_XS | 49.3* / 40.9 | 36.1* / 35.3 |
| Flint UD-IQ3_XXS / UD-Q4_K_XL | 42.2 / 16.7-19.1 (a pool + the SSD) | 41.0 / 20.8 (all on CPU: larger than Metal's limit) |
| Gemma 4 UD-IQ3_XXS / UD-Q4_K_XL | 28.8 / 24.0 | 26.8 / 24.1 |
| Lightning UD-IQ3_XXS / LFM2.5-24B-A2B UD-IQ3_XXS | 25.9 / 59.0 | 26.0 / 58.1 |
| **Mac mini** (8 GB, macOS 11) | | |
| Flint UD-IQ3_XXS | 8.7 | 2.96 (all on CPU: on the GPU it aborts) |
| Gemma 4 UD-IQ3_XXS | 8.4 (a 2.15 GiB pool) | 6.4 (30 on CPU) |
| LFM2.5-24B-A2B UD-IQ3_XXS | 11.7 | 8.8-10.6 (30-38 on CPU) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 26.0-28.4 | 43.8 (3 on CPU) |

In short:
- **Where the experts don't fit the GPU, MoMoE is well ahead.** On the PC that holds even against llama.cpp's MoE
  cache: Flash-Next 3x, Ornith-35B 2x, Ornith-122B 4x. MoMoE is also ahead with GLM on the Spark (13.3 against
  10.85 without drafting, 19.7 with it), LFM2.5-24B on the
  Jetson, and the three larger models on the Mac mini.
- **Where every expert fits in memory, the two are level.** That's Flint, Gemma 4 and Lightning on the Spark and the
  Macs, LFM2.5 on the Jetson, the M1 Ultra and the M4 Air, LFM2.5-8B on the PC, and Qwen3.6 on the M1 Ultra without
  drafting.
- **llama.cpp is ahead in a few places.** It computes some experts on the CPU from RAM while the GPU holds the rest,
  which wins for Flint IQ3_XXS and LFM2.5-24B on the PC and for LFM2.5-8B on the Mac mini. Its drafting gains more
  for Qwen3.6 on the M1 Ultra.

How each was measured (contexts, prompts, runs, llama.cpp builds) and the per-model notes are in
[design/benchmark-notes.md](design/benchmark-notes.md).

### What makes the engines fast

Each with the number it was measured at; details and switches in [design/benchmark-notes.md](design/benchmark-notes.md#what-the-engines-do-for-speed).

- **The CPU computes missing experts while the GPU computes its own** (`momoe-cuda`), instead of copying them over
  PCIe. Ornith-122B on the PC: 19.2 tok/s, against 11.4 with the copies.
- **Kept graphs**, built once per shape and replayed, as CUDA graphs on NVIDIA. LFM2.5-8B: 120 -> 185 tok/s on the
  M1 Ultra, 255 -> 305.5 on the PC.
- **A wait-for-event thread on Macs**, so macOS doesn't park the engine on efficiency cores at a low GPU clock.
  LFM2.5-8B through the server: 64-69 -> 120-128 tok/s.
- **Fewer copies:** the delta net's state is written in place and the router's choice is used as it is. Qwen3.6 on
  Metal: 76.5 -> 92.2 tok/s (llama.cpp 90.3).
- **Drafting's embedding rows looked up on the CPU**, not the whole table copied to the GPU per draft. Qwen3.6 on the
  Spark: 59.5 -> 125.2 tok/s.

## Install

You need:
- an NVIDIA card, RTX 20 series or newer, or an Apple Silicon Mac;
- enough RAM for the model;
- Python 3 (3.10 or newer on a Mac: `brew install python`);
- a C++ compiler and, with NVIDIA, the CUDA toolkit (on a Mac: `xcode-select --install`). Setup checks your machine
  and offers to install what is missing.

```sh
git clone https://github.com/aflin/momoe.git
cd momoe
./momoe setup                         # check the machine, choose a model, compile the engine, download, start
sudo ln -s "$PWD/momoe" /usr/local/bin/momoe    # optional: run momoe from any folder
```

Setup asks for a name for the model's config (for example `ornith-8gb`) and starts the server in the background.
`./setup.sh --check` lists what fits your PC. The Qwen3.6-family files are described in
[docs/QWEN36.md](docs/QWEN36.md) and [docs/QWEN35MOE_GGUFS.md](docs/QWEN35MOE_GGUFS.md).

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
momoe remove ornith-8gb               # delete a model: its GGUF, its pack, the config and its logs (asks first)
```

- **`momoe run`** starts a server if none runs, asking which config if there are several. It gives the tool its
  settings for that run only and installs nothing: a missing tool is named, with how to install it.
- **`momoe remove`** (or `delete`) lists what it will delete and asks first (`-y`: no question; `--dry-run`: only the
  list).
  - A model or pack that other configs also use stays, and it asks whether those go too (`--wipe`: they do, without
    asking).
  - It touches nothing outside `models/`, `packs/`, `config/` and `logs/`.
  - Given a folder in `models/` (a download without a config, say), it deletes that folder and its pack.
- **Pictures** (off unless asked for): `momoe setup --vision yes` downloads the model's image encoder and builds it,
  for every family that has one (all but Flint, Lightning and LFM2.5). Pictures work in a message and in a tool result
  (Claude Code's Read of an image). The encoder runs on the GPU on a Mac and a DGX Spark (1-3 s a picture) and on the
  CPU with a separate card (1-2 s; ~9 s for the Qwen3.6 family, which needs 1,024 tokens a picture to read text
  right). Checked on the RTX 4070 Ti (against llama.cpp's answers), the M1 Ultra and the Spark. On a Mac the encoder's
  memory comes out of the expert pool.
- **Other apps:** any OpenAI- or Anthropic-compatible client can use `http://127.0.0.1:<port>/v1`; `momoe status`
  shows the port. The server listens on 127.0.0.1 only. To reach it from another device, give `--host` together with
  `--api-key`, for example `momoe start ornith-8gb --host 0.0.0.0 --api-key KEY`. They apply to that start only; add
  `--save` to keep them in the config.

### Logs

Every log is in `logs/`:
- **`logs/<config>.log`** is the engine's: loading, the expert cache, one block of numbers per request. The engine
  logs keep growing.
- **`logs/server.log`** is the server's, started again with each `momoe start`. It writes a line a second while it
  answers, and a `done:` line per answer with its speed and the expert cache's hit rate.

```sh
momoe log                             # the last 20 lines of the server's and the engine's logs, then follow them
momoe log server                      # one log: server, engine, unsloth or lmstudio (all: the four)
momoe log -n 100                      # start from the last 100 lines
momoe log --no-follow                 # print the last lines and stop
```

## Two cards

Two NVIDIA cards can share a model in two ways. Setup writes either with `--gpus a,b` (and `--helper-tier`), or by
hand in the config: both cards in `"gpu"`, the main card first.

- **The layer split** (`"gpu": [a, b]`, `"layer_split": "auto"` or the first layer of each later card): each card runs
  its own layers. Setup splits GLM-5.3-Flash this way by itself across two cards under 16 GB each. Best for two equal
  cards (not measured).
- **The helper tier** (`"args": [..., "--helper-tier"]`): the second card is a second level of expert cache under the
  first.
  1. The main card runs the model as it would alone.
  2. The helper card holds the next experts in line and computes them itself.
  3. The RAM holds the rest.

  Each expert is in one place only: what the main card lets go moves down to the helper, and the helper's evictions
  move to RAM. **The PC keeps much more memory free**, at about the same speed. `--helper-gib G` limits the helper's
  VRAM for experts; every setting it turns on can be changed on its own
  ([docs/SECOND_GPU.md](docs/SECOND_GPU.md#the-helper-tier---helper-tier)).

Measured with Flash-Next IQ3_XXS: RTX 4070 Ti (main) + RTX 3060 12 GB (helper, also driving the desktop), 64K context,
`tools/strata_bench.py`, three runs per test, 2026-10-18.

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

Writing speed is about the same (the agent session varied 64.5-71.1 between runs of the same configuration); short
prompts are read faster and long ones slower. The answers are not bit-identical to one card's.

With Ornith-1.5-122B the helper tier also works with the CPU computing the experts neither card holds, but it is
slower: 12.0-12.2 tok/s against 19-21 on the 4070 Ti alone. More in
[design/benchmark-notes.md](design/benchmark-notes.md#the-two-card-measurements).

## Where things are

| Folder | What |
| --- | --- |
| `config/` | the model configs setup writes (yours; not in git) |
| `models/`, `packs/`, `mtp/` | the downloaded models and what setup prepares from them (not in git) |
| `engine/` | Strata's compiled engine (`momoe`, `momoe-vision`) |
| `build-cuda/`, `build-metal/` | MoMoE's compiled engines (`momoe-cuda`, `momoe-metal`) |
| `build-vision-momoe*/` | their image encoder (`momoe-vision`), when pictures are on |
| `logs/` | every log (above) |
| `.momoe/` | the server's state (which config runs) |
| `design/` | MoMoE's design notes: what was measured, chosen or rejected |
| `bench/results/` | the measurements behind the tables, with the scripts |
| `docs/` | Strata's documentation. Most of it applies; it says "Strata" where MoMoE has its own name |

## Credits

- **MoMoE**: its CUDA and Metal engines (`momoe-cuda`, `momoe-metal`: [src/metal](src/metal/README.md)), built on ggml,
  and the changes to Strata's engine, server and installer (the Qwen3.6 family's path began as
  [#1402](https://github.com/Niko1221/Strata/pull/1402)).
- **[Strata](https://github.com/Niko1221/Strata)**, by Niko1221 and the Strata contributors: the engine `momoe` runs
  (Strata's, with MoMoE's additions), and the server, the installer, the documentation and the paper MoMoE's grew from.
- Changes taken from other Strata forks and open pull requests, credited to their authors:
  - [Asofwar/Strata](https://github.com/Asofwar/Strata) by Asofwar: the Qwen3.6 decode-path speedups, the fast
    256 x 8 router on CUDA and the control-vector fix (three commits, under their own name).
  - [#2674](https://github.com/Niko1221/Strata/pull/2674) by Scott Page ([page1](https://github.com/page1)): the
    server logs a reply that ended after its thinking with no text and no tool call.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**, by the ggml authors: the ggml library under
  `momoe-cuda` and `momoe-metal` (in [src/metal/ggml](src/metal/ggml), with MoMoE's additions listed in its
  `README.momoe`), the model code their graphs for Nemotron, GLM-5.3, LFM2.5, Flint and Gemma 4 follow, and the GGUF
  format, the quantization kernels and the image encoder Strata builds on.
- The models' authors and quantizers listed above: the Qwen team, ISTA-DASLab, ukisai, Unsloth, bartowski, Ornith AI,
  Myric, NVIDIA, Z.ai, Liquid AI, Recursive and Google.

## License

MIT: see [LICENSE](LICENSE), which holds Strata's copyright notice and MoMoE's own. The copy of ggml in
[src/metal/ggml](src/metal/ggml) keeps its own MIT license (the ggml authors), beside it.
