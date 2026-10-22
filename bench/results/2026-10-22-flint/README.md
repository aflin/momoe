# 2026-10-22: Flint 40B-A2.5B on every machine

Recursive's Flint (unsloth/flint-GGUF at 9167d02e): 49 blocks of GQA attention with attention sinks (35 with a
513-token sliding window and RoPE, 14 full without RoPE), 256 experts and a shared one, 2.5B active; llama.cpp added it
in a37cc46 (#30928). MoMoE runs it on its ggml engine as an lfm2moe variant (src/metal: `flint`), with:

- the sliding-window blocks' keys and values in a ring of 1,792 positions (the window and the largest batch), read
  only as far as written until it wraps, saved whole at the conversation cache's checkpoints;
- llama.cpp's per-expert power-of-two scaling of the down projection's input (on CUDA; on Metal for batches above
  8 tokens only: its matrix-vector kernels read f32);
- on a separate card, the CPU computing the experts the card does not hold, through the mailboxes, in a decode graph
  kept and replayed as a CUDA graph, with the shared expert computed while the CPU works;
- the `flint` pre-tokenizer (three patterns one after another; the same tokens as llama.cpp's /tokenize on 14 texts);
- `momoe setup --family flint`: UD-Q4_K_XL where the GPU shares 48 GB or more (or a card of 16 GB or more, not
  measured), UD-IQ3_XXS on a 12 GB card and a 32 GB Mac.

Checked against llama.cpp a37cc46 (its greedy reference with top-20 log-probabilities, made on the PC): KL over 300
tokens fed its own, and its greedy tokens. Speed: the engine with every option as setup writes it, 32K context, the
second half of 300 tokens; llama.cpp: `llama-bench` tg128 at its best fit. `stats.csv` has every number; the scripts
are here (`t1.sh` on the PC, `zf.sh` on the M4 Air, `llref-ff.sh` / `llref-long.sh` for the references).

| Machine | Size | MoMoE | llama.cpp |
| --- | --- | ---: | ---: |
| PC (RTX 4070 Ti) | UD-IQ3_XXS | 88.2-89.9 | 102.6 |
| PC (RTX 4070 Ti) | UD-Q4_K_XL | 65.6-65.7 | 58.5 |
| DGX Spark | UD-Q4_K_XL | 72.9-73.2 | 76.0 |
| M1 Ultra | UD-Q4_K_XL | 84.2-84.4 | 77.4-77.8 |
| M4 Air | UD-IQ3_XXS | 42.2-42.8 | 41.0 |
| M4 Air | UD-Q4_K_XL | 16.7-19.1 | 20.8 |

Not done: the server checked on the PC only (chat, tool call whole and streamed, tool result); the Jetson.
