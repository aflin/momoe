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
| M4 Air | MacBook Air M4, 32 GB; "as an 8 GB Mac": 10 GiB wired away (`mlock`), about 5 GB left available |

## Which model runs where (writing speed, tok/s)

| Model, size | PC (4070 Ti) | PC, two cards | Spark | Jetson | M1 Ultra | M4 Air |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next IQ3_XXS | 57.9* | – | 31.8 | – | 28.1 | 7.5 (setup's 14 GiB pool; 8.4 over a 1,673-token server chat) |
| Flash-Next Coder IQ1_M | 54.8* | – | 28.4 | – | 25.5 ⁱ | – |
| Swift 1.5 IQ3_XXS | 63.4* | – | 30.3 | – | 25.7 ⁱ | – |
| Flash-Next (Unsloth) UD-IQ4_XS | 12.4* | – | 24.5 | – | 12.0 ⁱ | – |
| Qwen3.6-35B-A3B (Unsloth) UD-IQ4_XS | 158.7* | – | 61.0* | – | 110.8* | – |
| Qwen3.6-35B-A3B (bartowski) IQ4_XS | 157.5* | – | 77.1* | – | 96.4* | – |
| Ornith-1.5 IQ4_XS | 99.0* | – | 68.2* | – | 67.5 | 31.1* (as a 16 GB Mac, earlier: 9.7–13.1) |
| KAT-Coder i-quality-v2 | 126.2* | – | 57.3* | – | 82.5* | – |
| Nemotron 3.5 Bolt UD-IQ3_XXS | 29.7 | runs (split 18.0, helper 14.4, right after a start) | 27.4 | – | 33.7 (UD-Q3_K_XL) | – |
| GLM-5.3-Flash UD-IQ3_XXS | doesn't fit one 12 GB card | 0.5 (split) | 18.0* | – | 4.0* | – |
| LFM2.5-24B-A2B | 75.9 (UD-Q4_K_XL) | 42.1 (helper, store warm; 40.6 over 300 tokens from a start, against 20.4 on one card) | 126.0 | 6.9 | 70.5 | 12.1 (as an 8 GB Mac) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 255 | 197 (split) | 157.1 | 16.6 | 154.5 | 41.5 (as an 8 GB Mac) |
| Flash-Next NVFP4 | PC only (Strata's own packs), not tested in MoMoE | | | | | |

\* With drafting: the model's own multi-token-prediction (MTP) layer proposes up to 3 or 4 tokens ahead and the model
checks them in one pass (Strata's engine on the PC: `--spec 4`; `momoe-cuda` / `momoe-metal`: `--mtp 3` for GLM,
`--mtp 4` for the Qwen3.6 family). Without \*: no drafting. The Flash-Next family on the
Spark and the Macs and Ornith on the M1 Ultra were measured without it, the two-card runs too, LFM has no MTP
layer, and MoMoE does not yet draft with Bolt's. llama.cpp's numbers are all without drafting.

ⁱ Measured while macOS had moved the engine, started in the background over SSH, to the M1 Ultra's efficiency cores
(graph building ~4x slower: 3.5 against 0.8 ms a token; the same build wrote 154.5 tok/s on LFM2.5-8B before and 65
then, while llama.cpp wrote 190): likely about half of what it does when started in the foreground. The other M1 Ultra
numbers from 11:00-13:30 may be affected too; the LFM2.5-8B and Ornith ones (before 11:00) are not. A background
service would need launchd settings that keep it from being demoted.

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
| LFM2.5-8B UD-IQ3_XXS | Jetson | 18.7 | 20.5 |
| LFM2.5-24B UD-Q4_K_XL | PC | 75.9 | 105.1 |
| LFM2.5-8B UD-Q4_K_XL | PC | 255 | 305.8 |
| LFM2.5-8B UD-Q4_K_XL | M1 Ultra | 154.5 | 169 (190 later the same day) |
| LFM2.5-24B UD-IQ3_XXS | M1 Ultra | 70.5 | 108.7 |
| Qwen3.6 (Unsloth) UD-IQ4_XS | M1 Ultra | 110.8* | 90.2–91.2 |
| Qwen3.6 (bartowski) IQ4_XS | M1 Ultra | 96.4* | 96.8 |
| KAT-Coder i-quality-v2 | M1 Ultra | 82.5* | 87.5 |
| Bolt UD-Q3_K_XL | M1 Ultra | 33.7 | 38.5 (it cannot read prompts there: no Metal kernel for 9 experts a token) |
| Bolt UD-IQ3_XXS | PC | 30.0 / 30.6 (server, earlier) | 26.2 / 26.1 |
| GLM-5.3 UD-IQ3_XXS | Spark | 18.0* / 15.0* / 14.6* (server, 10-17); 13.3 (10-17) | 1.3 (10-17, a 48-token check with every expert on the CPU, cold; not a speed test) |
| GLM-5.3 UD-IQ4_XS | Spark | 11.7* / 10.4* (server, 10-17); 9.3 (10-17) | 2.3 / 2.5 |

\* With drafting (see above); the others without.

llama.cpp: `llama-bench` tg128 (b11724; b11699 on the PC, pinned to the 4070 Ti), or `llama-server` where marked. The
files larger than Metal's working set (48 GiB on the M1 Ultra) were not run in llama.cpp there.

Prompt reading on the PC (Strata's engine, 4K / 16K tokens): Flash-Next IQ3_XXS 1,178 / 2,133 tok/s; the Coder 1,931 /
2,615; Swift 1,196 / 2,163; Unsloth's UD-IQ4_XS 67 / 91; Ornith 3,001 / 4,357; Qwen3.6 (Unsloth) 3,767 / 4,502;
Qwen3.6 (bartowski) 3,536 / 4,504; KAT-Coder 3,081 / 3,974.
