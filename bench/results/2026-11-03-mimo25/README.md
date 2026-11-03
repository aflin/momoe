# 2026-11-03: Xiaomi MiMo-V2.5 on the PC

MiMo-V2.5 (Xiaomi, April 2026; Unsloth's GGUFs at f7aff786, UD-IQ2_XXS, 96.5 GB in three files): ~310B parameters,
~15B active. 48 blocks: 8 full-attention and 40 sliding-window (128 tokens), GQA with 4 / 8 KV heads, keys 192 wide and
values 128, RoPE on 64 of the key's dimensions (base 10M full, 10K window), attention sinks on the window blocks, the
attention output scaled by 0.707; 1 dense lead block and 47 MoE blocks of 256 experts (8 a token, sigmoid router with a
bias); 3 MTP blocks (not run). llama.cpp's `mimo2`; MoMoE runs it on its ggml engine as a flint variant (src/metal:
`mimo`).

The PC: RTX 4070 Ti 12 GB, Ryzen 9 5950X, 63 GB of RAM, the model on NVMe RAID 1. The model is larger than the RAM: most
experts are computed by the CPU (the trial at load picks that) and read from the disk as tokens need them.

## Against llama.cpp

llama.cpp 7182ef7 (`bench/results/2026-10-26-ling3/llref.sh`: its greedy answer with the top-20 log-probabilities, the
experts on its CPU, the rest on the GPU), MoMoE fed llama.cpp's tokens:

| Prompt | KL mean / p99 | Same top token |
| --- | --- | ---: |
| code (48 tokens, 199 steps) | 0.0006 / 0.0097 | 99.0% |
| medical (59 tokens, 199 steps) | 0.0043-0.0054 / 0.048-0.079 | 96.0-97.0% |
| a document (1,531 tokens, 199 steps), chunks of 1024 / 300 | 0.0056 / 0.0995, 0.0064 / 0.1083 | 98.0% |

The larger KL on two prompts is the CPU's arithmetic against the GPU's at 2 bits, not the port: llama.cpp entirely on
its CPU (`-ngl 0`) is KL 0.0044 from its own GPU run on the medical prompt's first step and picks the other token of
the same near-tie (0.52 / 0.47); MoMoE's first step is KL 0.0003 from that CPU run. Greedy, on its own picks, MoMoE's
answer is llama.cpp's (GPU) for the first 37 tokens of the code prompt, 1 of the medical one (llama.cpp CPU: also 1;
MoMoE against llama.cpp CPU: 82) and 21 of the document. The server's tokenizer (`qwen2` pre-tokenizer) and chat
template gave llama.cpp's prompt tokens for all three. Chat and an OpenAI tool call through the server were checked.

## Speed on the PC

The server's benchmark (`tools/strata_bench.py`, 3 runs, `server/`), `--ctx 32768 --chunk 1024`, the pool 4.1 GiB:
chat writing 5.9 tok/s (5.3-6.2), first token 10.5 s; the benchmark's 4K and 16K prompts 128 and 204 tok/s (repetitive
text: few experts); the agent session 4.5 tok/s writing and 43-101 s to its first token (35 tok/s reading real text).
llama.cpp (the reference runs above): 5.7-6.4 tok/s writing.

The prompt is read from the disk: each chunk touches most of a block's 256 experts. The engine alone on a 5,419-token
document: chunks of 1024 34.1 tok/s (444 GiB read), 2048 65.8 tok/s (239 GiB read); 4096 does not fit the 12 GB card at
32K context.
