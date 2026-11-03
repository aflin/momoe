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
