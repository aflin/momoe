# 2026-11-03: Xiaomi MiMo-V2.6-Flash-MOPD on the PC

MiMo-V2.6-Flash-MOPD (Xiaomi, 2026-09-27): MiMo-V2.5's architecture and shapes (`mimo2`: 48 blocks, 256 experts, 8 a
token; see [2026-11-03-mimo25](../2026-11-03-mimo25)) trained further; MOPD is its last training stage (its card: it also
stops the model repeating the same tool calls). Its experts are MXFP4 in the original. AesSedai's MoE quants at
b6e95d58, BPW2.5 (96.7 GB in three files): the experts per layer from IQ1_M to IQ3_XXS, everything else Q6_K / Q8_0;
the repo's own measurement against the MXFP4 original: perplexity +10.5%, KL 0.227 (BPW3.0 +5.5%, BPW2.0 +27%). The
same engine path as V2.5 (src/metal: `mimo`), no code changes; its file has 7 format groups (V2.5's: 3).

The PC: RTX 4070 Ti 12 GB, Ryzen 9 5950X, 63 GB of RAM, the file on NVMe. Its tokenizer is V2.5's (`qwen2`); the chat
template is new (no default system prompt; tool calls `<tool_call><function=...><parameter=...>` without line breaks).

## Against llama.cpp

llama.cpp 7182ef7 (its experts on the CPU), MoMoE fed llama.cpp's tokens, 199 steps:

| Prompt | KL mean / p99 | Same top token |
| --- | --- | ---: |
| code (31 tokens) | 0.0015 / 0.0519 | 100% |
| medical (42 tokens) | 0.0106 / 0.2287 (the GPU computing every expert: 0.0116; the CPU most of them: 0.0079) | 96.5% |
| a document (1,514 tokens) | 0.0079 / 0.1045 | 97.5% |

Larger than V2.5's (0.0006 / ~0.005 / 0.0056): the CPU's arithmetic against the GPU's on this mix of 1- and 2-bit
experts. MoMoE against itself, the GPU computing every expert against the CPU computing most of them, on the medical
prompt: KL 0.0067 over the whole vocabulary, the size of its difference from llama.cpp. Greedy, on its own picks, the
code prompt's 200 tokens are llama.cpp's. The server's tokens equal llama.cpp's for all three prompts.

## The server on the PC

Setup's config (`--ctx 32768 --chunk 1024`, firefly's installed engine): a short chat, an OpenAI tool call
(`get_weather({"city": "Paris", "unit": "C"})`) and then the 5,402-token document (read at 32.8 tok/s, no VRAM
warning; the pool 2.4 GiB). The engine alone on the code prompt: 5.9 tok/s writing (6.2 the second half), 26% of the
experts it used in the card's pool, the rest computed by the CPU from the file cache and the SSD. llama.cpp's runs
above: 5.1-5.8 tok/s.

The server's benchmark (`tools/strata_bench.py`, 3 runs, `server-pc/`), the same config, ~40 GB of RAM available:

| | median (range) |
| --- | --- |
| chat, writing | 5.4 tok/s (4.9-6.5) |
| chat, first token | 9.0 s (8.0-10.6) |
| the 4K / 16K prompt, reading | 151 / 230 tok/s (repetitive text: few experts) |
| agent session, writing | 4.2 tok/s (3.3-4.5) |
| agent turn, first token | 79.5 s (45.1-103.6; ~3,000 new tokens each) |

MiMo-V2.5's the same way: 5.9 / 4.5 tok/s writing (chat / agent), 74 s to an agent turn's first token.

## The M1 Ultra (64 GB)

The engine alone, setup's pool (25 GiB, 30% of the experts), the code prompt against llama.cpp's file: KL 0.0012, p99
0.0272, top-1 100%; 7.4-7.5 tok/s writing, 91.6% of the experts each token used in the pool, the rest read from the
SSD.

The server's benchmark (`tools/strata_bench.py`, 3 runs, `server-m1-ultra/`), setup's Mac config (`--ctx 32768
--pool-gib 25 --nocache`), nothing else running:

| | median (range) |
| --- | --- |
| chat, writing | 7.0 tok/s (5.7-7.8) |
| chat, first token | 4.8 s (4.6-5.3) |
| the 4K / 16K prompt, reading | 166 / 202 tok/s (repetitive text: few experts) |
| agent session, writing | 5.6 tok/s (5.2-5.9) |
| agent turn, first token | 63.6 s (32.0-75.1; ~3,000 new tokens each) |

Writing is near DeepSeek-V4.1-Flash's 7.4 there; reading real text is the slow part, as on the PC.
