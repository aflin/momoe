# Pictures on MoMoE's engines

**Status:** Implemented (52432e88, 2026-10-23). Opt-in: `momoe setup --vision yes`; off by default on every engine.

## What it covers

| Engine | Families with pictures | Where checked |
| --- | --- | --- |
| Strata's engine (PC) | Flash-Next (IQ3_XXS, Coder, Swift, Unsloth), Qwen3.6 (Unsloth, bartowski), Ornith-1.5-35B, KAT-Coder | RTX 4070 Ti |
| `momoe-cuda` | Ornith-1.5-122B, Gemma 4, Nemotron 3.5 Bolt, GLM-5.3-Flash; on the Spark also every Qwen family | RTX 4070 Ti, DGX Spark |
| `momoe-metal` | the same families | M1 Ultra (64 GB) |

No image encoder is published for Flint, Nemotron 3.5 Lightning or LFM2.5: setup does not ask for them.

Not tested with pictures: the M4 Air, the 8 GB Mac mini and the Jetson (updated, not run). GLM-5.3-Flash on the RTX
4070 Ti alone does not start (9.8 GiB of dense weights); its PC config needs the second card.

Strata's engine already read pictures (inherited from Strata); MoMoE only had to build its encoder and download the
files. The rest of this note is about MoMoE's own engines and the server.

## How it works

- **The encoder** is `tools/vision` (llama.cpp's mtmd), a separate process the server starts. It writes each picture's
  rows (`SVE1` records: n, nx, ny, n_embd, then the rows). For a model without M-RoPE (Gemma 4) it writes nx = n,
  ny = 1.
- **The request** is `GENI <max_new> [key=value ...] <file> <ids>`, as for Strata's engine. The engine puts each record's
  rows on the prompt's next run of image cells (`<|image_pad|>`, Gemma 4's and GLM-5's `<|image|>`, Bolt's `<image>`).
- **The graph:** `x = x * keep + img` after the token embeddings (and after Gemma's scaling: llama.cpp does not scale
  image rows). Only graphs whose tokens include image cells have the two extra inputs.
- **Positions:** Qwen's M-RoPE grid as mtmd lays it out: a picture's cells at (t, h, w) = (q, q + y, q + x), the text
  after it at q + max(nx, ny). Also for the MTP layer. Gemma 4 and GLM-5 number picture cells as text.
- **Gemma 4 26B-A4B:** a picture's cells see each other in both attention masks (llama.cpp's
  `mtmd_decode_use_non_causal`), so a picture is never split across prompt chunks, and one larger than the batch is
  refused. For other models a picture may span chunks (Bolt's encoder made 3,239 rows for a wide diagram, above the
  300-token cap).
- **The caches:** in the key the conversation caches are matched by, a picture's cells are negative numbers made from
  its rows (FNV-1a), so a cached prompt is reused only up to the first picture that differs. The same picture again
  reuses it (M1 Ultra, Ornith-35B: 1.8 s, then 0.7 s).
- **The server:** the marker rules per template (`_image_marks`); a blank line after a Gemma picture; Gemma 4 with
  thinking off goes through `UnaskedThought` (below).

## Measurements and the choices they made

- **Where the encoder runs.** On a CPU without BF16 instructions the BF16 encoder files are slow: the M1 Ultra 50-60 s
  a picture, the Spark's Grace CPU 30-50 s (one picture there passed the server's 300 s limit while a build ran). On
  their GPUs: 1.8 s (Metal) and 1.7-2.7 s (CUDA). With a separate card the encoder stays on the CPU (Ryzen 9 5950X:
  1-2.5 s at 300 tokens), so the card keeps its memory for the experts. setup: Metal on a Mac, CUDA on a GPU that shares
  the system's memory, the CPU otherwise.
- **The Mac's memory.** The encoder's process peaks at its file plus 0.15-0.2 GB (0.91 GB file: 1.06-1.09 GB; Bolt's
  1.73 GB: 1.93 GB), but the engine beside it needs more room than that. GLM-5.3-Flash UD-IQ3_XXS on the M1 Ultra: a
  33 GiB pool without pictures, with them 30 runs and 31 runs out of Metal's memory. Swift IQ3_XXS with every expert
  in memory ran out at its first picture. setup now counts the encoder file plus 2 GiB as fixed weights in the pool fit
  (GLM: 30 GiB; Swift: a 34 GiB pool instead of every expert, so its text is slower with pictures on; not measured
  how much).
- **Tokens per picture.** At 300 tokens (setup's CPU default) the Qwen3.6 architecture's models (Qwen3.6, Ornith-1.5 35B
  and 122B, KAT-Coder) misread a 2x2 grid of words; so did llama.cpp with the same files and 300 tokens. llama.cpp's
  encoder warns that Qwen-VL needs at least 1,024. With min and max 1,024 all five read both grids, in a message and in
  a tool result, on the 4070 Ti, the Spark and the M1 Ultra. Cost on the 5950X's CPU encoder: ~9 s a picture against
  1-2. The Flash-Next families, Gemma 4, Bolt and GLM read the grids at 300.
- **Gemma 4 and a picture in a tool result.** Its template ends the prompt right after the picture there. UD-Q4_K_XL
  then wrote junk before its answer (`weight: 0.0`, `图-2x2 grid ...`) or looped; UD-IQ3_XXS did not. A blank line after
  `<image|>` (as Gemma 3's processor writes) made both clean on the 4070 Ti and the Spark, and left the answers to
  pictures in a message the same.
- **Gemma 4 with thinking off** sometimes writes its whole answer in a thought block it never closes (the picture in a
  tool result, M1 Ultra, Q4_K_XL): the reply came back empty. `UnaskedThought` holds that text; it is reasoning when an
  answer or a call follows, the answer when the reply stops inside it.

## Checked against llama.cpp

On the RTX 4070 Ti, llama.cpp 7182ef7's `llama-server` with the same GGUF and mmproj, `--image-max-tokens 300`: the same
prompt token counts and, on the clear pictures, the same answers (Ornith-122B, Gemma 4, Bolt; Ornith-35B on Strata's
engine). Where the model was unsure (the word grid at 300 tokens) both engines misread it, in different ways.

## Text output unchanged

Greedy, the engine alone (`--tokens`, 64 tokens), two runs each: `main` and the change gave the same tokens with
Gemma 4 and Ornith-1.5-122B. Through the server, text is not repeatable from run to run even with `main` alone (the
expert path trial and the pool's contents depend on timing), so server runs cannot show a regression.

## The encoder's llama.cpp

The encoder is built from llama.cpp 7182ef7 (`VISION_LLAMA_COMMIT`), newer than the engine's pinned copy: only its mtmd
reads GLM-5's (`glm5v`) and Nemotron 3.5's (`nemotron_omni`) encoders. Strata's engine keeps its own encoder from the
pinned copy.

## Not done

- A GPU encoder for a separate NVIDIA card on MoMoE's engine (it would take VRAM from the experts).
- Pictures on the M4 Air, the 8 GB Mac and the Jetson: not run. On an 8 GB machine the encoder's ~1-2 GB may not fit
  beside the model.
- Gemma 4's audio and video markers: not handled.
