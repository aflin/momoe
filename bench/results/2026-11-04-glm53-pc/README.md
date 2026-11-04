# 2026-11-04: GLM-5.3-Flash UD-IQ2_XXS on the PC

GLM-5.3-Flash (Z.ai, 320B-A18B; `glm5-next`: KDA + sparse MLA, 45 blocks, 288 experts, 8 a token, one MTP block),
Unsloth's UD-IQ2_XXS at a38483c8 (101.8 GB in four files): its experts 86.2 GiB, its other weights 5.9 GiB with
the head (the token embeddings stay in RAM), one layer's experts 2.1 GiB (the staging copy prompts need). Before this,
setup's only sizes were UD-IQ3_XXS and larger (6.8 GiB of other weights, 2.4 GiB a layer): "does not fit a 12 GB
card".

The PC: RTX 4070 Ti 12 GB + RTX 3060 12 GB (it also drives the desktop), Ryzen 9 5950X, 63 GB of RAM, the file on
NVMe (`/`: 10.8 GB/s of random 4 MiB reads, fio, 16 threads). The file is larger than the RAM: about half of what a
token's experts need comes from the SSD.

## What changed in the engine

- **The CPU computes the experts the cards lack** (as for Ling-3.0-flash, whose MoE layer GLM shares): it was left
  out of `hybrid_model`. The engine's trial chose it every time: one card 273 ms a token against 414 with the copies,
  two cards 234 against 304. Not on the DGX Spark (its memory is the GPU's: the pool there, as measured).
- **The pool's floor at 4 tokens' experts** (`MOMOE_POOL_MIN_TOKENS=4`): with the default 16 the first pool (3.1
  GiB) left the 4070 Ti no room for the staging copy and it did not start.

## Against llama.cpp

llama.cpp 7182ef7, one slot, every expert on the CPU (`-ngl 999 --n-cpu-moe 99`), the 4070 Ti; MoMoE fed its tokens,
199 steps (`ref-*.prompt`, `ref-*.tok`, `llref.sh`):

| Prompt | 4070 Ti alone | both cards (layer split) | llama.cpp writing |
| --- | --- | --- | ---: |
| code (36 tokens) | KL 0.0025 / p99 0.0606, 99.5%; 3.8 tok/s | 0.0022 / 0.0499, 100%; 4.5 tok/s | 4.0 tok/s |
| medical (45 tokens) | - | 0.0075 / 0.0974, 98.5%; 5.8 tok/s | 5.4 tok/s |

The medical prompt with the GPU computing every expert (the copies, `MOMOE_QWEN_HYBRID=0`): KL 0.0105, 96.5%, 3.9
tok/s. The CPU path is the closer one (llama.cpp computes these experts on the CPU too): the rest is the GPU's
arithmetic on 1-2-bit experts, as with MiMo.

## One card or two

`--layer-split auto` puts blocks 0-23 on the 4070 Ti and 24-44 on the 3060; the pools then hold 9.2 GiB (11% of the
experts) against 1.9 GiB (2%) on the 4070 Ti alone, and a token's chosen experts are found there 36% of the time
against 17%. Setup splits by itself across two cards under 16 GB each; on one 12 GB card it starts, with the warning
that it wants about 16 GB.

**Drafting** with its MTP block (`--mtp 3`, setup's default for the family) does not pay on separate cards: 4.4
tok/s against 4.5 without, though 76% of the drafts were kept (3.1 tokens a window): a window of 4 tokens took 786 ms
against 222 for one, since each token's experts are read or computed for it. It also took 2.5 GiB from the 3060's
pool. Setup leaves it out on separate cards (`card_no_draft`).

## The server

Setup's config on the two cards (`--ctx 65536 --chunk 1024 --layer-split auto`, `MOMOE_POOL_MIN_TOKENS=4`, no
drafting): a chat with its thinking and an OpenAI tool call (`get_weather({"city": "Paris", "unit": "C"})`), then its
benchmark (`tools/strata_bench.py`, 3 runs, `server-pc/`):

| | median (range) |
| --- | --- |
| chat, writing | 4.7 tok/s (4.4-5.4) |
| chat, first token | 7.5 s (6.8-9.9) |
| the 4K / 16K prompt, reading | 82 / 122 tok/s (repetitive text: few experts) |
| agent session, writing | 4.0 tok/s (3.4-4.3) |
| agent turn, first token | 84.4 s (50.6-97.7; ~3,000 new tokens each) |

Not measured: UD-IQ2_XXS on the Spark and the Macs (setup keeps UD-IQ3_XXS there), llama.cpp with both cards.
