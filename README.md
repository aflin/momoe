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
- **MoMoE's ggml engine** (`momoe-cuda`), on Linux with an NVIDIA card, including the DGX Spark, whose GB10 shares its
  128 GB of memory with the CPU. The experts that don't fit are read from the SSD directly, at 10-13 GB/s. It runs
  Nemotron 3.5 Bolt and GLM-5.3-Flash.
- **MoMoE's Metal engine** (`momoe-metal`, [src/metal](src/metal/README.md)), on Apple Silicon Macs (early). On a Mac
  with less than about 28 GB, the experts that don't fit are read from the SSD as needed.

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
| **M4 Air** | MacBook Air M4, 32 GB |
| **M4 Air, simulated 16 GB** | the same MacBook with 16 GiB of its memory wired away (`mlock`), so that macOS can neither use nor swap it |
| **M1 Ultra** | Mac Studio M1 Ultra, 64 GB |

Speeds are in tokens per second. "Writes" is the speed of the answer, "reads" the speed of reading the prompt.
"Answered" means the model was set up there and answered, but its speed hasn't been measured.

| Model | Sizes setup offers | Tested on | Writes | Reads | llama.cpp writes, same machine |
| --- | --- | --- | ---: | ---: | ---: |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), files by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S (about 48-62 GB of RAM) | PC, IQ3_XXS, 64K context | 61 | 2,170 (32K tokens) | |
| | | M4 Air, IQ3_XXS, 12 GiB of experts in memory, drafting on (set up by hand) | 8.6-8.9 | 56-62 (1K-8K tokens) | 4.2 (CPU only: on the GPU it runs out of memory) |
| | | M1 Ultra, IQ3_XXS, every expert in memory (set up by hand; with drafting too it does not fit the GPU's memory limit) | 26.7 | 214 (8K tokens, 34 GiB in memory) | 34.0 |
| Flash-Next Coder, files by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | IQ1_M (about 32 GB of RAM) | PC | answered | | |
| Swift 1.5, a Flash-Next fine-tune, files by [ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | IQ2_XS, IQ3_XXS, IQ3_S | not tested in MoMoE | | | |
| Flash-Next, files by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | UD-Q4_K_XL, UD-IQ4_XS (part of the experts read from the SSD) | not tested in MoMoE | | | |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), files by [Unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) and [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) | UD-IQ4_XS, UD-IQ3_S and 24 more (8-12 GB cards, 16-32 GB of RAM) | PC | answered | | |
| [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a Qwen3.6 fine-tune, files by [bartowski](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) | IQ4_XS, IQ3_XXS | PC, IQ4_XS, 262K context | 96-109 | about 4,130 (24K tokens, 32K context) | |
| | | M4 Air, IQ4_XS, every expert in memory, drafting on (what setup installs on a Mac) | 34.5 | | 33.7 |
| | | M4 Air, simulated 16 GB, IQ4_XS, setup's 16 GB config (5 GiB of experts in memory, 32K context, drafting on) | 9.7-13.1 | 126 (9,698 tokens) | |
| | | M1 Ultra, IQ4_XS, every expert in memory | 54.6 | 582 (8K tokens, measured before every expert was preloaded) | 88.9 |
| KAT-Coder-V2.5-Dev, a coding fine-tune of Qwen3.6, files by [Myric](https://huggingface.co/Myric/KAT-Coder-V2.5-Dev-MTP-APEX-GGUF) | i-quality-v2, i-quality | PC | answered | | |
| [Nemotron 3.5 Bolt 74B-A9B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16), files by [Unsloth](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF) (`--family bolt`) | UD-IQ3_XXS, UD-Q3_K_XL (Linux and NVIDIA only; about 56 GB of RAM) | PC, UD-IQ3_XXS | 30.0 / 30.6 | | 26.2 / 26.1 |
| | | Spark, UD-Q4_K_XL (not offered by setup), every expert in memory | 20.2 | | |
| [GLM-5.3-Flash 320B-A18B](https://huggingface.co/zai-org/GLM-5.3-Flash), files by [Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (`--family glm53`) | UD-IQ3_XXS (setup's choice), UD-IQ4_XS, UD-Q3_K_XL (the DGX Spark; text only) | Spark, UD-IQ3_XXS, drafting on | 15.0 / 14.6 | | not measured |
| | | Spark, UD-IQ4_XS, drafting on | 11.7 / 10.4 | 69 (3,620 tokens) | 2.3 / 2.5 |

How these were measured:
- **Two numbers ("30.0 / 30.6"):** the server, 400-token chats, greedy / sampled.
- **Flash-Next on the PC:** medians of three runs of `tools/strata_bench.py`.
- **Ornith on the PC:** short chats.
- **Ornith on the simulated 16 GB Mac:** the server, three 400-token chats (greedy); the same as with all 32 GB.
- **Bolt on the Spark:** the engine on its own, not the server.
- **Macs:** the details are in [design/mac-port.md](design/mac-port.md).
- **llama.cpp:**
  - on the M1 Ultra, `llama-server` with six 500-token answers;
  - on the Spark, the experts read from its mapped file (`-cmoe --no-repack`; its default settings ran the machine out
    of memory). It read GLM's 3,620-token prompt at 4.1 tokens/s.

GLM-5.3-Flash's UD-IQ3_XXS (121 GB) writes faster than UD-IQ4_XS (157 GB) on the Spark because the expert store holds
nearly every expert it uses; `--model UD-IQ4_XS` installs the larger file.

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

## Where things are

| Folder | What |
| --- | --- |
| `config/` | the model configs setup writes (yours; not in git) |
| `models/`, `packs/`, `mtp/` | the downloaded models and what setup prepares from them (not in git) |
| `engine/` | the compiled engine (`momoe`, `momoe-vision`) |
| `.momoe/` | the server's state and log |
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
