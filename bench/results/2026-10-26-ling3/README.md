# 2026-10-26: Ling-3.0-flash Sante on the PC and the M1 Ultra

inclusionAI's Ling-3.0-flash-Sante (the healthcare fine-tune of Ling-3.0-flash; bartowski's GGUFs at d59f495f, IQ3_XXS,
55.7 GB in two shards): 124B parameters, 5.1B active; 35 KDA blocks and 7 MLA blocks (RoPE on 64 of the key's 192
dimensions, a sigmoid gate per head), 512 experts in 8 groups (top-8 of the 4 best groups) and a shared one, 2 dense
lead blocks, one MTP block. llama.cpp's `bailingmoe3`; MoMoE runs it on its ggml engine beside glm5-next
(src/metal: `ling`).

Checked against llama.cpp 7182ef7 on the PC (its greedy answer with the top-20 log-probabilities, the experts on its CPU:
`llref.sh`; the `.prompt` and `.tok` files here, the `.kl` files are not kept): KL over the answer fed llama.cpp's own
tokens, and the greedy tokens. Speed: the engine as setup writes it, the server's benchmark (`tools/strata_bench.py`,
3 runs each; `server/`).

## Against llama.cpp

| Prompt | Machine | KL mean / p99 | Same top token | Greedy tokens the same |
| --- | --- | --- | ---: | ---: |
| code (44 tokens, 299 steps) | PC, copies | 0.0005 / 0.0073 | 99.0% | 299 of 299 |
| code | PC, the CPU's experts | 0.0005 / 0.0071 | 99.0% | 299 of 299 |
| code | M1 Ultra (an 8 GiB pool) | 0.0004 / 0.0072 | 99.7% | 299 of 299 |
| medical (55 tokens, 199 steps) | PC, copies / the CPU's experts | 0.0013 / 0.0144, 0.0013 / 0.0146 | 99.5% | 199 of 199 |
| medical | M1 Ultra | 0.0013 / 0.0173 | 100% | 199 of 199 |
| a document (1,589 tokens, 199 steps), chunks of 1024 / 300 | PC, copies | 0.0014 / 0.0249, 0.0013 / 0.0145 | 98.0%, 99.0% | 199 of 199 |
| a document | PC, the CPU's experts | 0.0012 / 0.0099 | 98.5% | 199 of 199 |
| a document | M1 Ultra | 0.0012 / 0.0130 | 99.0% | 199 of 199 |

MoMoE's tokenizer (the `bailingmoe2` pre-tokenizer) and the chat template gave llama.cpp's tokens for all three
prompts. Verify windows of 4 gave the same next tokens as windows of 1.

## Speed on the PC (RTX 4070 Ti 12 GB, Ryzen 9 5950X, 63 GB of RAM; 64K context)

| | the copies, chunk 1024 (the first version) | the CPU's experts, kept graph, chunk 2048 | chunk 4096 (setup's) |
| --- | ---: | ---: | ---: |
| short chat, writing (tok/s) | 25.7 | 34.1 | 33.5 |
| agent session, writing | 24.3 | 31.8 | 38.2 |
| 4K prompt, reading | 744 | 1,007 | 1,189 |
| 16K prompt, reading | 882 | 1,192 | 1,494 |
| agent session, first token | 9.79 s | 6.68 s | 4.28 s |

The engine alone (299 tokens of the code prompt, two runs each): the copies 24.3-24.8 tok/s, the CPU computing the
experts the pool lacks 33.6-33.7, with the decode graph kept and replayed as a CUDA graph 36.5-36.8. The trial at load
chose the CPU's experts every time (28-29 against 44-48 ms per token). MOMOE_FILLS 16-256: all within 33.5-34.3.
llama.cpp (`llama-server`, every expert on its CPU, `--n-cpu-moe 99`): 14.3 tok/s on the code prompt; not its best
placement, which was not searched.

Drafting with the MTP block is correct (greedy, it parts only at near-ties: a smaller expert pool parts at the same
token without drafting) but slower on this card: 28.0 (3 drafts) and 29.9 (2) against 36.7; setup leaves it out.

## The plain model (Ling-3.0-flash IQ3_XXS, bartowski's GGUFs at 7e1a1243)

Against llama.cpp (the code prompt, 299 steps): KL mean 0.0009, p99 0.0156, the same top token 99.7%, 299 of 299 greedy
tokens the same. Speed with setup's arguments (64K context, chunk 4096): 35.3-35.6 tok/s (the second half), Sante
33.5 the same way. The server's chat and tool calls checked.

## llama.cpp's best on the PC

Its server at 32K context, the code prompt, 300 tokens greedy, three runs (the first warms its cache): MoE cache 6 GiB
(`--cpu-moe --moe-cache-mib 6144`) 9.5 / 27.1 / 27.4 tok/s; `--n-cpu-moe 38` 13.8 / 26.0 / 26.7, 40 11.0 / 25.7 / 24.9;
36, and a cache of 7 or 8 GiB, did not fit. `llama-bench` tg128: `--n-cpu-moe` 36 / 38 / 40 23.6 / 22.9 / 22.2.

## M1 Ultra (64 GB, macOS; setup's config: a 38 GiB expert pool, the rest read from the SSD, 64K context, --mtp 3)

| | the server's benchmark |
| --- | ---: |
| short chat, writing (tok/s) | 23.0 |
| agent session, writing | 19.6 |
| 4K prompt, reading | 620 |
| 16K prompt, reading | 548 |
| agent session, first token | 8.87 s |

The engine alone (299 tokens of the code prompt): a 34 or 38 GiB pool (97.6% of the experts found there) 16.7 tok/s
without drafting, 45.5 of its 58.8 ms per token waiting at the layers' pauses; drafting spreads them over a window:
25.0 / 25.7 / 24.6 tok/s with 2 / 3 / 4 drafts (2.5-2.8 tokens a window). With the pool fitted to Metal's working set
(41.8 GiB) the GPU ran out of memory; setup's 38 GiB runs. An 8 GiB pool beside another model server: 14.7 tok/s.
