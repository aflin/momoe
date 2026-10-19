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
| **M4 Air as an 8 GB Mac** | the same MacBook with 10 GiB of its memory wired away (`mlock`), which left about 5 GB available, as on an 8 GB Mac |

### Which model runs where

Writing speed in tokens per second; "-" means not tried. Every model here was installed with `momoe setup` on the
machine; its greedy output was compared with llama.cpp's (48 or 64 tokens: the same, or parting only where the top two
tokens are within a few hundredths of each other, which was checked); and the server answered a chat, made a tool call
(whole and streamed) and answered after the tool's result.

| Model, size | PC | Spark | Jetson | M1 Ultra | M4 Air |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next IQ3_XXS | 57.9* | 31.8 | - | 28.1 | 7.5 |
| Flash-Next Coder IQ1_M | 54.8* | 28.4 | - | 25.5 ⁱ | - |
| Swift 1.5 IQ3_XXS | 63.4* | 30.3 | - | 25.7 ⁱ | - |
| Flash-Next (Unsloth) UD-IQ4_XS | 12.4* | 24.5 | - | 12.0 ⁱ | - |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 158.7* | 61.0* | - | 110.8* | - |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 157.5* | 77.1* | - | 96.4* | - |
| Ornith-1.5-35B-A3B IQ4_XS | 99.0* | 68.2* | - | 67.5 | 31.1* (as a 16 GB Mac: 9.7-13.1) |
| KAT-Coder-V2.5-Dev i-quality-v2 | 126.2* | 57.3* | - | 82.5* | - |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 29.7 | 27.4 | - | 33.7 (UD-Q3_K_XL) | - |
| GLM-5.3-Flash UD-IQ3_XXS | does not fit a 12 GB card | 18.0* | - | 4.0* | - |
| LFM2.5-24B-A2B | 75.9 (UD-Q4_K_XL) | 126.0 | 6.9 | 70.5 | 12.1 (as an 8 GB Mac) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 255 | 157.1 | 16.6 | 154.5 | 41.5 (as an 8 GB Mac) |
| Flash-Next NVFP4 | Strata's own packs: a PC with an NVIDIA card only, not tested in MoMoE | | | | |

\* With drafting: the model's own multi-token-prediction (MTP) layer proposes up to 3 or 4 tokens ahead and the model
checks them in one pass, keeping only what it would have written itself. Numbers without \* are without drafting:
the Flash-Next family on the Spark and the Macs and Ornith on the M1 Ultra were measured without it, LFM has no
MTP layer, and MoMoE does not yet draft with Bolt's. llama.cpp's numbers are all without drafting.

ⁱ Measured while macOS had moved the engine, started in the background over SSH, to the efficiency cores (graph
building ~4x slower; the same build wrote 154.5 tok/s on LFM2.5-8B before and 65 then): likely about half of what it
does when started in the foreground. A background service would need launchd settings that keep it from being demoted.

Which engine: on the PC, Strata's engine for the Qwen3.8-Flash-Next and Qwen3.6 families (Ornith with MoMoE's ggml
engine: 63.9) and MoMoE's ggml engine (`momoe-cuda`) for Bolt, GLM and LFM; `momoe-cuda` for everything on the Spark
and the Jetson; `momoe-metal` on the Macs.

### Against llama.cpp on the same machine

| Model | Machine | MoMoE | llama.cpp |
| --- | --- | ---: | ---: |
| LFM2.5-24B-A2B UD-IQ3_XXS | Jetson | 6.9 (engine) / 7.0 (server) | 4.8 (16 layers' experts on the GPU, the most that fit) / 3.7-4.3 (server) |
| LFM2.5-8B-A1B UD-Q4_K_XL | Jetson | 16.6 | 17.2 (18 of 24 layers' experts on the GPU) |
| LFM2.5-8B-A1B UD-IQ3_XXS | Jetson | 18.7 | 20.5 (all on the GPU) |
| LFM2.5-24B-A2B UD-Q4_K_XL | PC | 75.9 | 105.1 (10 layers' experts on the CPU, the best fit) |
| LFM2.5-8B-A1B UD-Q4_K_XL | PC | 255 | 305.8 (all on the card) |
| LFM2.5-8B-A1B UD-Q4_K_XL | M1 Ultra | 154.5 | 169 |
| LFM2.5-24B-A2B UD-IQ3_XXS | M1 Ultra | 70.5 | 108.7 |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | M1 Ultra | 110.8* | 90.2-91.2 |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | M1 Ultra | 96.4* | 96.8 |
| KAT-Coder-V2.5-Dev i-quality-v2 | M1 Ultra | 82.5* | 87.5 |
| Nemotron 3.5 Bolt UD-Q3_K_XL | M1 Ultra | 33.7 | 38.5 (it cannot read a prompt: its Metal build has no kernel for 9 experts a token) |
| Nemotron 3.5 Bolt UD-IQ3_XXS | PC | 30.0 / 30.6 (server, greedy / sampled) | 26.2 / 26.1 |
| GLM-5.3-Flash UD-IQ3_XXS | Spark | 18.0* / 15.0* / 14.6* (server, greedy / sampled); 13.3 | 1.3 (a 48-token check: every expert on the CPU, cold, not a speed test) |
| GLM-5.3-Flash UD-IQ4_XS | Spark | 11.7* / 10.4* (server); 9.3 | 2.3 / 2.5 |

\* With drafting (see above); the others without.

Where every expert fits on the GPU (the 8B on the PC; most models on the M1 Ultra), llama.cpp is ahead: MoMoE builds
its graph again for every token (up to a fifth of a token's time on the M1 Ultra) and adds a few small kernels a
layer. MoMoE gains where the experts do not fit: the Jetson, GLM on the Spark, Bolt on the PC.

Reading prompts on the PC, with Strata's engine (4K / 16K tokens): Flash-Next IQ3_XXS 1,178 / 2,133 tok/s; the Coder
1,931 / 2,615; Swift 1,196 / 2,163; Unsloth's UD-IQ4_XS 67 / 91; the Qwen3.6 family 3,001-3,767 / 3,974-4,504.

How these were measured (2026-10-19 unless named; build `momoe-cuda` / `momoe-metal` with ggml from llama.cpp b11724):
- **Strata's engine on the PC:** `tools/strata_bench.py`, three runs, medians: "chat decode" in the table.
- **`momoe-cuda` and `momoe-metal`:** the engine on its own, greedy, the second half of a 300-400-token answer, with
  the arguments setup wrote (drafting where they have it); "server" rows: the server's own speed over a long chat.
- **The Spark:** one model at a time under a watchdog. GLM-5.3-Flash with `--mtp 3`; its server and the runs without
  drafting, 2026-10-17. UD-IQ3_XXS (113 GiB, setup's choice on the Spark) is a quarter smaller than UD-IQ4_XS (146
  GiB): nearly all its experts fit in memory (97.6% of expert reads from the pool) and it writes 1.3-1.4 times as
  fast, more with drafting.
- **The Jetson:** 16K context. llama.cpp b11724, `llama-bench` tg128.
- **The Macs:** setup's configs: every expert in memory where it fits beside the KV cache (Flash-Next IQ3_XXS and Swift
  at 32K context, the others at 64K), else an expert pool and the rest from the SSD (Unsloth's Flash-Next: 39 GiB on the
  M1 Ultra; GLM: 29 GiB; Flash-Next IQ3_XXS on the M4 Air: 14 GiB, from the memory free when setup ran). The files larger than Metal's working set (48 GiB on the M1 Ultra) were not run in llama.cpp:
  that took a MacBook down. The 8 GB-Mac rows: setup's 8 GB config (a 3 GiB pool, 16K context); with all 32 GB the
  same runs wrote 42.8 and 14.8 tok/s. It does not simulate an 8 GB Mac's smaller Metal limit.
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
