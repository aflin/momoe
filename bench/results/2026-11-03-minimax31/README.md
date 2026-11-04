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
and the compute buffers), so setup splits the layers across two (`--layer-split auto`: 0-30 on the 4070 Ti, 31-59 on
the 3060) with 512-token chunks, `MOMOE_POOL_MIN_TOKENS=4` and a 1.5 GiB VRAM margin (cuBLAS's own buffers). The
pools hold 2% of the experts; the rest are read from the file cache and the SSD as tokens need them.

The engine alone: 1.4-1.7 tok/s writing (77-81% of the experts a token used in a pool), prompts read at 8-12 tok/s.
**llama.cpp is faster here**: 2.8-3.1 tok/s writing (every expert on the CPU), because MoMoE's CPU-expert path is off
with a layer split, so every miss is copied to a card. Prompts: MoMoE 12.0 tok/s on the 6,246-token one, llama.cpp
6.2.

The server, setup's config (`--ctx 32768 --chunk 512`): a chat, an OpenAI tool call and a 4K-token prompt; the
server's benchmark was not run at this speed.

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

Not done: the Spark (TBD), MiniMax's MTP layer, the CPU computing experts beside a layer split.
