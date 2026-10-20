# Every model on every machine, 2026-10-19

Which MoMoE model runs where, and how fast it writes, measured on one day across the machines below. Each cell with a
number was installed with `momoe setup` on that machine. Its greedy output was compared with llama.cpp's (48 or 64
tokens): the same, or parting only at a near-tie that was checked. The server answered a chat, made a tool call (whole
and streamed) and answered after the tool's result. Every row, with its conditions, is in [stats.csv](stats.csv).

| Name | Machine |
| --- | --- |
| PC | Linux (Debian 12), RTX 4070 Ti (12 GB), Ryzen 9 5950X, 63 GB of RAM |
| PC, two cards | the same PC with its RTX 3060 (12 GB, which also drives the desktop) beside the 4070 Ti |
| Spark | NVIDIA DGX Spark: GB10, 128 GB shared by the CPU and the GPU |
| Jetson | Jetson Orin Nano Super developer kit: 8 GB shared, 15 W mode, NVMe SSD |
| M1 Ultra | Mac Studio M1 Ultra, 64 GB |
| M4 Air | MacBook Air M4, 32 GB; "as an 8 GB Mac": 10 GiB wired away (`mlock`) |

## Which model runs where (writing speed, tok/s)

| Model, size | PC (4070 Ti) | PC, two cards | Spark | Jetson | M1 Ultra | M4 Air |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next IQ3_XXS | 57.9* | – | 35.5 | – | 38.2 | 7.9* (setup's 11 GiB pool; earlier engine, 14 GiB, no drafting: 7.5) |
| Flash-Next Coder IQ1_M | 54.8* | – | 33.3 | – | 48.9* (56.0* before the change below; two other prompts faster) | – |
| Swift 1.5 IQ3_XXS | 63.4* | – | 34.6 | – | 37.9 | – |
| Flash-Next (Unsloth) UD-IQ4_XS | 12.4* | – | 30.2 | – | 19.0* (36 GiB pool; server 17.9) | – |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 158.7* | – | 125.2* | – | 129.3* | – |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 157.5* | – | 123.6* | – | 117.1* | – |
| Ornith-1.5 IQ4_XS | 128.8* (setup's config; 99.0* with a 256K context) | – | 104.1* | – | 92.7* | 31.2* (14 GiB pool, 2026-10-20; 26.9* the day before; as a 16 GB Mac, earlier: 9.7–13.1) |
| KAT-Coder i-quality-v2 | 126.2* | – | 99.4* | – | 104.9* | – |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 29.7 | runs (split 18.0, helper 14.4, right after a start) | 28.4 | – | 37.5 (UD-Q3_K_XL) | – |
| GLM-5.3-Flash UD-IQ3_XXS | doesn't fit one 12 GB card | 0.5 (split) | 19.7* | – | 4.5* | – |
| LFM2.5-24B-A2B | 75.9 (UD-Q4_K_XL) | 42.1 (helper, store warm; 40.6 over 300 tokens from a start, against 20.4 on one card) | 134.0 | 6.9 | 104.3 (server 99.8) | 11.9 (as an 8 GB Mac; 12.0 with all 32 GB) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 305.5 | 197 (split) | 161.7 | 16.6 | 185.3 (server 177.5) | 43.1 (as an 8 GB Mac; 43.3 with all 32 GB) |
| Flash-Next NVFP4 | PC only (Strata's own packs), not tested in MoMoE | | | | | |

\* With drafting: the model's own multi-token-prediction (MTP) layer proposes up to 3 or 4 tokens ahead and the model
checks them in one pass (Strata's engine on the PC: `--spec 4`; `momoe-cuda` / `momoe-metal`: `--mtp 3` for GLM,
`--mtp 4` for the Qwen3.6 family). Without \*: no drafting. The Flash-Next family on the
Spark, and Flash-Next IQ3_XXS and Swift on the M1 Ultra (every expert fits only without the draft layer), were measured without it, the two-card runs too, LFM has no MTP
layer, and MoMoE does not yet draft with Bolt's. llama.cpp's numbers: with drafting where marked \* (2026-10-20), else without.

The Mac numbers were measured again in the evening with the engine that keeps one thread waiting on wait-for-event
while a request runs (see the README). The earlier ones (in stats.csv) were measured with macOS holding the engine on
the efficiency cores at random: graph building 0.8 or ~4.2 ms a token, LFM2.5-8B 150 or 62-68 tok/s run to run, and its
server 64-69 on every request.

The M1 Ultra column was measured again on 2026-10-20 with the engine that keeps its graphs (see the README; the same
tokens as before on all twelve models, with graphs kept, without, and with the previous engine). Unsloth's Flash-Next
and GLM keep their 2026-10-19 numbers: with an expert pool, a graph is still built each token.

The Spark column and LFM2.5-8B on the PC were measured again on 2026-10-20 with `momoe-cuda` keeping its graphs (run
as CUDA graphs) and looking up the drafting layer's token rows on the CPU (it had copied the whole embedding table to
the GPU for every drafted token); the same tokens as the day's earlier build on every model. GLM-5.3 on the Spark: 19.4 -> 19.7
(160 tokens, the same tokens; its drafting had no such copy).

The M1 Ultra's Qwen3.6 and Flash-Next families were measured once more after two `momoe-metal` changes (see the
README): the delta net's state written in place (ggml-metal's GATED_DELTA_NET + CPY fusion) and the router's choice
passed to the experts as it is. Without drafting, Qwen3.6 76.5 -> 92.2 (llama.cpp 90.3).

Engines: on the PC Strata's engine for the Qwen3.8-Flash-Next and Qwen3.6 families (`tools/strata_bench.py`, three
runs, the chat decode median; Ornith on `momoe-cuda`: 63.9) and `momoe-cuda` for Bolt, GLM and LFM; `momoe-cuda` for
everything on the Spark and the Jetson; `momoe-metal` on the Macs. `momoe-cuda` and `momoe-metal`: greedy, the second
half of a 300-400-token answer, with the arguments setup wrote (drafting where they have it); ggml from llama.cpp
b11724. The two-card column: GLM and LFM2.5-8B with the layer split (`"layer_split": "auto"`), LFM2.5-24B and Bolt with
`--helper-tier`; the split and the helper gave the same tokens as one card (LFM2.5-8B: 200 of 200; LFM2.5-24B with the
helper: 300 of 300; Bolt with the helper: 200 of 200, the split at a near-tie).

## Against llama.cpp on the same machine

| Model | Machine | MoMoE | llama.cpp |
| --- | --- | ---: | ---: |
| LFM2.5-24B UD-IQ3_XXS | Jetson | 6.9 / 7.0 (server) | 4.8 / 3.7–4.3 (server) |
| LFM2.5-8B UD-Q4_K_XL | Jetson | 16.6 | 17.2 |
| LFM2.5-8B UD-IQ3_XXS | Jetson | 20.3 (every expert in memory) | 20.5 |
| LFM2.5-24B UD-Q4_K_XL | PC | 75.9 | 105.1 |
| LFM2.5-8B UD-Q4_K_XL | PC | 305.5 | 305.8 |
| LFM2.5-8B UD-Q4_K_XL | M1 Ultra | 185.3 / 177.5 (server) | 169 (190 later the same day) / 176–182 (server) |
| LFM2.5-24B UD-IQ3_XXS | M1 Ultra | 104.3 / 99.8 (server) | 108.7 |
| Qwen3.6 (Unsloth) UD-IQ4_XS | M1 Ultra | 129.3* / 92.0 | 150.1* (4 drafted) / 90.3 |
| Qwen3.6 (bartowski) IQ4_XS | M1 Ultra | 117.1* / 95.8 | 146.1* (4) / 96.1 |
| KAT-Coder i-quality-v2 | M1 Ultra | 104.9* / 91.0 | 106.8* (3) / 89.6 |
| Ornith-1.5 IQ4_XS | M1 Ultra | 92.7* / 96.1 | 96.2* (2) / 96.1 |
| Bolt UD-Q3_K_XL | M1 Ultra | 37.5 | 38.5 (it cannot read prompts there: no Metal kernel for 9 experts a token) |
| Bolt UD-IQ3_XXS | PC | 30.0 / 30.6 (server, earlier) | 26.2 / 26.1 |
| GLM-5.3 UD-IQ3_XXS | Spark | 18.0* / 15.0* / 14.6* (server, 10-17); 13.3 (10-17) | 1.3 (10-17, a 48-token check with every expert on the CPU, cold; not a speed test) |
| GLM-5.3 UD-IQ4_XS | Spark | 11.7* / 10.4* (server, 10-17); 9.3 (10-17) | 2.3 / 2.5 |

\* With drafting (see above); the others without. llama.cpp drafts with the same MTP layer (`llama-server
--spec-type draft-mtp`, b11724), shown at its best number of drafted tokens (2-4); MoMoE drafts 4. Both sides for these
rows: greedy, the same prompt, 400 tokens (MoMoE: its engine, the answer's second half; llama.cpp: its server's own
timing over all 400). Without drafting the two are level (92.0-96.1 tok/s against 89.6-96.1); llama.cpp gains more
from drafting (Qwen3.6: 1.66x against 1.41x), the gap left. Ornith's MTP layer is a weak drafter in both (llama.cpp: 54%
of its drafts kept at 2, 28% at 4): without drafting both write 96.1. On the M4 Air, where it runs from a 14 GiB
pool and the SSD, drafting still wins (30.7-31.7 against 24.6-25.2 tok/s without: a window shares the experts' reads).

llama.cpp: `llama-bench` tg128 (b11724; b11699 on the PC, pinned to the 4070 Ti), or `llama-server` where marked. The
files larger than Metal's working set (48 GiB on the M1 Ultra) were not run in llama.cpp there.

Prompt reading on the PC (Strata's engine, 4K / 16K tokens): Flash-Next IQ3_XXS 1,178 / 2,133 tok/s; the Coder 1,931 /
2,615; Swift 1,196 / 2,163; Unsloth's UD-IQ4_XS 67 / 91; Ornith 3,679 / 4,521 (setup's config; 3,001 / 4,357 with a 256K context); Qwen3.6 (Unsloth) 3,767 / 4,502;
Qwen3.6 (bartowski) 3,536 / 4,504; KAT-Coder 3,081 / 3,974.
