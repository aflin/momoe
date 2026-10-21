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

Not tested: the Jetson (8 GB) with the 122B or the big Flash-Next files; Flash-Next NVFP4 (Strata's own packs).
