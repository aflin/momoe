# 2026-11-03: MiniMax M3.1 Flash on the PC (two cards) and the M1 Ultra

MiniMax M3.1 Flash (MiniMax, 428B-A23B; `minimax-m3` in its GGUF): 60 blocks, the first 3 with a dense FFN, the rest
128 experts (4 a token, a sigmoid router with a bias, scale 2.0) and one shared expert; the FFN is `swigluoai`
(alpha 1.702, limit 7). Its attention: 64 query heads and 4 KV heads of 128, RoPE on 64 of them, and MiniMax Sparse
Attention in every block: an indexer (4 heads of 128) scores the keys, the scores are max-pooled per 128-token block,
and each query attends to its own block and the 16 best others. Unsloth's UD-IQ2_XXS at b17b7f54 (135 GB in four
files): its experts 1-2 bit, 9.2 GB of other weights. Its license lets anyone use it commercially if the product says
"Built with MiniMax M3.1-Flash" (setup prints the terms before the download).

The engine path: src/metal (`mm3`, beside MiMo's), the server's MiniMax reply format (`<mm:think>`, its namespaced
tool calls; serve/test_minimax_calls.py), the `minimax-m2` pre-tokenizer.

## Against llama.cpp

llama.cpp 7182ef7 with a local patch (it rejects this file: 1,188 tensors against the 1,176 it expects, the indexer
tensors of the three dense blocks), one slot (`-np 1`: with more slots llama.cpp silently attends densely), its
experts on the CPU. MoMoE fed llama.cpp's tokens, 199 steps (`ref-*.prompt`, `ref-*.tok`):

| Prompt | PC (two cards) KL mean / p99, same top token | M1 Ultra |
| --- | --- | --- |
| code (215 tokens) | 0.0024 / 0.0328, 98.5% | 0.0014 / 0.0248, 98.5% |
| medical (224 tokens) | 0.0025 / 0.0363, 97.5% | 0.0018 / 0.0228, 99.0% |
| a document (1,658 tokens) | 0.0038 / 0.0802, 99.0% | - |
| a document (6,246 tokens: 49 blocks, so the selection matters) | 0.0070 / 0.2300, 98.0% | 0.0045 / 0.0751, 99.5% |

The short prompts fit in the 17 blocks a query may see, so they check everything but the selection; the 6,246-token
one checks it. One open point: the config marks all 60 blocks sparse and the file has indexers in the three dense
ones, but llama.cpp attends fully in those three. MoMoE follows the config (`MOMOE_MM3_LEAD_FULL=1` does llama.cpp's);
on the 6,246-token prompt on the PC: MoMoE llama.cpp's way against llama.cpp 0.0066, MoMoE's way against llama.cpp
0.0070, against llama.cpp patched to attend sparsely in all 60 blocks 0.0125, against dense attention 0.0326.

## The PC: RTX 4070 Ti 12 GB + RTX 3060 12 GB

Ryzen 9 5950X, 63 GB of RAM, the file on NVMe. One 12 GB card cannot hold it (9.2 GB of other weights, the KV cache
and the compute buffers), so setup splits the layers across two (`--layer-split auto`: 0-31 on the 4070 Ti, 32-59 on
the 3060, which also drives the desktop) with 512-token chunks, `MOMOE_POOL_MIN_TOKENS=4` and a 1.5 GiB VRAM margin
(cuBLAS's own buffers), and a 16K context: at 32K the KV cache needs 1.9 GiB more than the two cards have left. The
pools hold 1-2% of the experts.

**The CPU computes the experts the cards lack** (2026-11-04; before, with a layer split, every missing expert was
copied to a card): the engine's trial measured 400-420 ms a token on the CPU against 830-870 ms with the copies. The
engine alone, 199 tokens:

| Prompt | KL mean / p99, same top token | writing (second half) |
| --- | --- | ---: |
| code | 0.0020 / 0.0365, 99.0% | 2.9 tok/s (copies: 1.4) |
| medical | 0.0021 / 0.0295, 98.0% | 3.8 tok/s (copies: 1.7) |
| the 6,246-token document | 0.0076 / 0.2169, 98.0% | 3.3 tok/s (copies: 1.4) |

The same agreement with llama.cpp as with the copies (above), so the CPU's arithmetic adds no error of its own.
llama.cpp on this machine (every expert on the CPU): 2.8-3.1 tok/s writing; prompts: MoMoE 12 tok/s on the
6,246-token one, llama.cpp 6.2.

The server, setup's config (`--ctx 16384 --chunk 512`): a chat, an OpenAI tool call
(`get_weather({"city": "Paris", "unit": "C"})`), then its benchmark (`tools/strata_bench.py`, 3 runs, `server-pc/`):

| | median (range) |
| --- | --- |
| chat, writing | 2.8 tok/s (2.7-3.1) |
| chat, first token | 29.2 s (28.2-30.7) |
| the 4K prompt, reading | 22.7 tok/s (the 16K one: past the 16K context) |
| agent session, writing | 2.7 tok/s (2.5-3.0) |
| agent turn, first token | 233.5 s (113.2-242.0; ~3,000 new tokens each) |

Ling-3.0-flash and Ornith-1.5-122B on the same split, with the new default: their trials chose the copies, as
before (10.4 and 11.9 tok/s writing, 128 tokens); Nemotron keeps the copies beside a split (not measured).

## The M1 Ultra (64 GB)

Setup's Mac config (`--ctx 32768 --pool-gib 20 --nocache`): the pool 21.2 GiB, 18% of the experts, 84% of those a
token used found in it. The engine alone on the code prompt: 3.2 tok/s writing (the second half), the 6,246-token
prompt read at 29.1 tok/s. Setup's pool for this file on a 64 GB Mac would be 32 GiB, ~49 GiB of Metal memory with
the rest, past what is safe there (~40): `mac_margin_gib` 12 brings it to 20.

The server: a chat and an OpenAI tool call (`get_weather({"city": "Paris", "unit": "C"})`), then its benchmark
(`tools/strata_bench.py`, 3 runs, `server-m1-ultra/`), nothing else running:

| | median (range) |
| --- | --- |
| chat, writing | 2.7 tok/s (2.3-3.0) |
| chat, first token | 14.4 s (14.0-14.7) |
| the 4K / 16K prompt, reading | 58 / 69 tok/s (repetitive text: few experts) |
| agent session, writing | 2.6 tok/s (2.1-3.0) |
| agent turn, first token | 99.0 s (49.5-106.3; ~3,000 new tokens each) |

## Drafting (DSpark)

M3.1 Flash has no MTP layer: it comes with a DSpark drafter instead (the original's `dspark/`; Unsloth's
`dspark-MiniMax-M3.1-Flash-Q8_0.gguf`, 2.1 GB, at the same revision): five dense blocks that read the model's state
at five of its layers and propose up to 7 tokens, checked by the model in one pass. llama.cpp 7182ef7 and its master
of 2026-11-04 do not load it (`swigluoai` is not one of the activations their DFlash loader takes). Unsloth's card:
about 1.5x on code with one drafted token a check, on a GPU that holds the whole model.

Drafting pays only if a check of several tokens costs much less than that many single tokens. Here it does not:
the experts each token chooses are read or computed for that token, and two tokens share few of them. Teacher-forced
windows of the code prompt's 199 tokens (MOMOE_TF_WINDOW, every token accepted), ms a window:

| | 1 token | 2 tokens | 4 tokens |
| --- | ---: | ---: | ---: |
| PC (two cards, CPU experts) | 402 | 934 | 1,706 |
| M1 Ultra (a 20 GiB pool) | 340 | 585 | 997 |

On the PC a check of 2 costs more than 2 single tokens: drafting can only lose. On the M1 Ultra one drafted token a
check gains only if more than ~78% of the drafts are kept (585 ms plus the drafter's own time for 1 + kept tokens,
against 340 a token); the acceptance Unsloth's 1.5x implies (~50-60%) would give ~2.6 tok/s against 2.9 without,
and the drafter's 2.1 GB would come out of the pool. Not built for now; worth it on a machine whose memory holds
the whole model.

Not done: the Spark (TBD), the drafter.
