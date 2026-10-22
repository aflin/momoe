# 2026-10-22: an 8 GB M1 Mac mini on macOS 11 (Big Sur)

A Mac mini M1 with 8 GB of memory, macOS 11.7.10; its GPU's working set (`recommendedMaxWorkingSetSize`) is 5.33 GiB.
Every model installed with `momoe setup` (a 3 GiB expert pool, the rest read from the SSD, 16K context), measured,
then removed with `momoe delete`. MoMoE: the engine with the config's arguments, the second half of 300 tokens, two
runs. llama.cpp a37cc46, built there with Metal: `llama-bench` tg64 at the best of the splits tried (`-ngl 99
--n-cpu-moe N`, or the CPU alone). Correct output: KL against llama.cpp's reference (made on the PC) for Flint and
Gemma 4; for the LFM2.5 models MoMoE's greedy tokens equal the M4 Air's (macOS 26) for the same prompt.

| Model | MoMoE (setup's config) | llama.cpp (its best) |
| --- | ---: | ---: |
| Flint 40B-A2.5B UD-IQ3_XXS | 8.7-8.8 (KL 0.0022, top-1 98%) | 2.96 (the CPU alone: on the GPU it aborts, see below) |
| LFM2.5-8B-A1B UD-Q4_K_XL | 26.0-28.4 (a 4 GiB pool: 32.8) | 43.8 (`--n-cpu-moe 3`; 2 and fewer run out of GPU memory) |
| LFM2.5-24B-A2B UD-IQ3_XXS | 11.7 (a 4 GiB pool: 13.3) | 8.8-10.6 (`--n-cpu-moe 30-38`, from run to run) |
| Gemma 4 26B-A4B UD-IQ3_XXS | 8.4 (KL 0.0008, top-1 98.3%) | 6.4 (`--n-cpu-moe 30`) |

Where the model is larger than the memory, MoMoE is ahead (its pool and reads from the SSD); LFM2.5-8B (5 GB) fits
in memory, and llama.cpp, computing the experts the GPU does not hold on the CPU, is ahead (`momoe-metal` has no CPU
expert path: it reads them into the pool).

Found on macOS 11 and fixed (fcbb7f18 and the next commit):
- Metal's bfloat kernels exist only from MSL 3.1 (macOS 14): a BF16 matmul aborted (Flint's 47 BF16 routers). The
  engine loads BF16 weights as F32 where the GPU cannot run them; ggml-metal reports no bfloat before macOS 14.
- ggml-metal's flash-attention vec kernel (one token) computed wrong results (Flint: KL 7.6); not used before macOS 14.
- Gemma 4's dense weights (1.92 GiB) with setup's 3 GiB pool needed 6.45 GiB of a 5.33 GiB working set: the GPU's
  work failed and the engine read choices it never wrote (a crash). `momoe-metal` now reserves the largest graph's
  buffers at load and shrinks the pool to the working set (Gemma 4: 2.15 GiB; Flint 3.00 -> 2.92, 8.8 -> 8.7 tok/s).

llama.cpp on the same machine: Flint on the GPU aborts (the missing bfloat kernel; with `GGML_METAL_BF16_DISABLE=1`,
`GGML_ASSERT(buf_dst)`); its vec flash-attention kernel is the same as MoMoE's was (not checked how its output fares).
Unsloth Studio does not install on macOS 11: current Unsloth needs MLX 0.32.4, which needs macOS 14.

Not tested: the server on the three models other than Flint; larger pools than setup's for Flint and Gemma 4.
