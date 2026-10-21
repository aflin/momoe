# 2026-10-21: every model on every machine

What was missing from the chart in the main README: Ornith-1.5-122B-A10B on all four machines, and eight models
that had never run on the MacBook Air M4 (32 GB). Each was installed with `momoe setup --family NAME --yes` on the
machine (setup chose the size, pool, context and drafting; the downloads went through setup), then:

1. the engine with setup's config, 300 tokens (GLM 100), speed of the second half;
2. the engine fed llama.cpp's own greedy tokens, without drafting: KL from llama.cpp's top-20 log-probabilities each
   step (`MOMOE_KL_REF`), and how often the top token is the same;
3. the first 64 greedy tokens against llama.cpp's;
4. the server (`momoe start`): a 400-token chat, a tool call (whole and streamed), the answer after the tool's result.

The reference: llama.cpp c0fd786 on the DGX Spark (CUDA; GLM on its CPU, its CUDA build pins the mapped file), the
same token ids for both, the model's own chat template. Where the greedy tokens part, the KL shows the engines agree
as closely as where they do not (a near-tie flips). `stats.csv` has every number; the scripts are here:
`llref2.sh` (the reference, on the Spark), `ztest.sh` (steps 1-4, on the Mac), `srvcheck.py` (step 4).

Found and fixed on the way (07c8e590): Nemotron's read-ahead on Metal went to the layer after a MoE block, not the
next MoE block (0% used; it filled a 282-slot group and the engine stopped); GLM's drafting from a 9% pool ran Metal
out of memory and was slower (setup now leaves it off below 15%).

## Against llama.cpp on the M4 Air

llama.cpp 99db719 (the M1 Ultra's build), every expert in memory (32 GB, nothing locked away), the model's chat
template rendered by llama-server and the same token ids given to MoMoE, greedy 400 tokens, two runs each, 90 s apart
(the Air has no fan). llama.cpp: its server's own timings over the 400 tokens (`--spec-type draft-mtp N`); MoMoE: the
engine with setup's config (`momoe setup --config-name ornith-32gb.json`, `lfm25-32gb.json`), the second half.
`zll.sh` ran it.

| Model | llama.cpp, no drafting | llama.cpp, drafting | MoMoE, no drafting | MoMoE, setup's config |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B UD-IQ4_XS | 28.9 | 43.8 / 46.8 / 47.6 (2 / 3 / 4 drafted) | 33.2 | 60.8 (`--mtp 4`) |
| Ornith-1.5-35B-A3B IQ4_XS | 35.3 | 36.1 / 32.0 / 27.3 | 40.9 | 49.3 (`--mtp 4`) |
| LFM2.5-24B-A2B UD-IQ3_XXS | 58.1 | (no MTP layer) | 58.5-59.0 | 58.8-59.0 |

llama.cpp's draft acceptance: Qwen3.6 0.89 / 0.82 / 0.77, Ornith 0.58 / 0.40 / 0.30. Swap stayed at 0.6 GB; the
lowest memory available 4.3 GB (Ornith drafting).

Not tested: the Jetson (8 GB) with the 122B or the big Flash-Next files; Flash-Next NVFP4 (Strata's own packs).
