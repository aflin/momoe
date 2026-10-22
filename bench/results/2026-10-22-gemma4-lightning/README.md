# 2026-10-22: Gemma 4 26B-A4B and Nemotron 3.5 Lightning 30B-A3B on every machine

Two new families on MoMoE's ggml engines (`momoe-cuda`, `momoe-metal`), as llama.cpp a37cc46 runs them:

- **Gemma 4 26B-A4B** (unsloth/gemma-4-26B-A4B-it-GGUF at c099eb48, `gemma4`): 30 blocks of attention, 25 with a
  1,024-token sliding window (head size 256, 8 KV heads; their keys and values in a ring, as Flint's) and 5 full ones
  (head size 512, 2 KV heads, no value projection, proportional RoPE); a dense GELU FFN beside a softmax-routed MoE of
  128 GELU experts (gate and up in one tensor, each expert's output scaled), 4B active; the logits soft-capped. Its
  tokenizer is SentencePiece-style BPE (`gemma4`: spaces as U+2581, byte tokens); its chat format marks the thinking
  with `<|channel>thought ... <channel|>` and the calls with `<|tool_call>call:NAME{...}<tool_call|>`, which the
  server rewrites into its own (serve/frontend.py GemmaRewriter).
- **Nemotron 3.5 Lightning 30B-A3B** (unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF at f2d3fe36,
  `nemotron_h_moe`): Bolt's Mamba-2 + attention + MoE without the latent projections, 128 experts, 3B active.

Bringing Lightning up found three faults in momoe-cuda that other models could meet too:

- CUDA's quantized batch kernels read each row on to a multiple of 512 values (multiplied by zeros). In the staging
  copy one group's experts sat right after another group's, so an expert's last row read the other group's bytes as
  Q8_0 scales: NaN (Lightning's down rows are 1,856 values). Each staged expert now has a padding row, kept zero.
- With the CPU computing the missing experts, every missing expert is the one zero slot, so a token can use that slot
  several times. ggml-cuda's batch path (MMQ, batches of 9 to 16 tokens) assumes an expert at most once per token: the
  ID map left entries unset (an illegal memory access), and only one token's worth of columns was computed for that
  expert (wrong tokens). Momoe marks these matmuls (`MOMOE_MMID_REPEATS`); ggml-cuda then maps every use and lets an
  expert have every column. Unmarked matmuls are computed as before.
- After a prompt batch that went through the pool (17-31 tokens), Nemotron's mailbox graphs read slots that had been
  given to other experts, because the stale-table rebuild ran for qwen35moe and Flint only. It runs for Nemotron too.

Checked against llama.cpp a37cc46: its greedy reference with top-20 log-probabilities (made on the PC, `llref-c.sh`),
KL over 299 tokens fed its own, and its greedy tokens. Speed: the second half of 300 tokens, twice (`tm.sh` on the PC,
`tmo.sh` on the Spark, `tmm.sh` on the Macs); llama.cpp: `llama-bench` tg128 at its best fit (on the PC
`--n-cpu-moe`; for Gemma 4 IQ3_XXS also llama-server at a 32K context, where everything no longer fits). `stats.csv` has
every number.

| Machine | Model | Size | MoMoE | llama.cpp |
| --- | --- | --- | ---: | ---: |
| PC (RTX 4070 Ti) | Gemma 4 | UD-IQ3_XXS | 89.4-92.4 (32K context) | 90.5 at 32K (`-ncmoe 4`); 131.5 at tg128's ~128 |
| PC (RTX 4070 Ti) | Gemma 4 | UD-Q4_K_XL | 74.6-75.2 | 54.8 (`-ncmoe 11`) |
| PC (RTX 4070 Ti) | Lightning | UD-IQ3_XXS | 70.9-71.1 | 65.3 (`-ncmoe 24`) |
| PC (RTX 4070 Ti) | Lightning | UD-Q4_K_XL | 37.7-38.7 | 40.2 (`-ncmoe 36`) |
| DGX Spark | Gemma 4 | UD-Q4_K_XL | 58.5-59.3 | 57.5 |
| DGX Spark | Lightning | UD-Q4_K_XL | 66.8-66.9 | 69.6 |
| M1 Ultra | Gemma 4 | UD-Q4_K_XL | 84.9-85.2 | 86.7 |
| M1 Ultra | Lightning | UD-Q4_K_XL | 89.9 | 92.1 |
| M4 Air | Gemma 4 | UD-IQ3_XXS | 27.9-28.8 | 26.8 |
| M4 Air | Gemma 4 | UD-Q4_K_XL | 23.6-24.0 | 24.1 |
| M4 Air | Lightning | UD-IQ3_XXS | 25.7-25.9 | 26.0 |

KL (mean; top-1 agreement): Gemma 4 0.0005-0.0008 (98.3-100%), all 299 greedy tokens equal on the PC (CPU experts) and
the Spark; Lightning 0.0048-0.0078 (98.3-99.3%), as much without the CPU's experts (0.0068 on the PC): its reference
computed 45 of its MoE layers on llama.cpp's CPU.

`momoe setup --family gemma4` and `--family lightning` picked UD-IQ3_XXS on the PC (12 GB card); `momoe start` served
each: Gemma 4 74.2-74.9 tok/s overall in a 400-token chat, Lightning 54.9-55.6 (warm); for both a tool call whole,
streamed and after its result; Gemma 4's thinking with `enable_thinking` and with a reasoning effort. Setup and the
server were run on the PC only. Gemma 4's tokens equal llama.cpp's /tokenize on 33 texts, with and without special
tokens (`tkcmp.py`).

Not changed by these fixes (`ab.sh`, the committed engine against this one on the PC): Ornith-1.5-35B IQ4_XS and Flint
UD-Q4_K_XL the same tokens and speed; Bolt UD-Q4_K_XL the same speed (15.0-16.1 against 15.6-16.0), its tokens part
at 63 of 200 (where its experts are computed moved: KL against its reference 0.0086-0.0094 before, 0.0088-0.0118
after, top-1 99.0% both).

Not tested: AMD, other cards, a 16-24 GB card (setup's 4-bit rule there is not measured), Gemma 4's images.
