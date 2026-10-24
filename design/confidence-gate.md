# A confidence gate: a low-bit model for every word, a higher-bit one only where it is unsure

**Status:** Rejected for models that fit in memory (2026-10-24). Open for a model several times larger than memory,
where the higher-bit file runs mostly from the SSD (DeepSeek-V4 Flash on the Spark or a 64 GB Mac, GLM-5.3-Flash
UD-IQ4_XS on the Spark): not measured there.

## The idea

PROMETHEVS (github.com/Magnvs7/Promethevs, "compress-then-correct") runs DeepSeek-V4.1-Flash from its full 510 GB on a
64 GB PC with an RTX 3060: every word is computed on a 2-bit copy of the experts, and a word is computed again with the
exact weights when the cheap pass's top two candidates are close (their logit gap below 0.45, "the gate"). Its 2-bit
copy alone picked the full model's word 97.4% of the time, 98.3% with the corrections; 4.4-7.4 tok/s against 2.14 for
the exact model on that PC. It needs 534 GiB of derived files (its own 2-bit copies, "correction planes", FP4 copies),
which MoMoE's rule (the engine runs the files as downloaded) rules out.

The version without derived files: two GGUF sizes of the same model, both downloaded as they are, a low-bit one for
every word and a 4-bit one for the words the gate flags.

## What was measured

Qwen3.6-35B-A3B on the RTX 4070 Ti, llama.cpp 7182ef7 (`llama-server`, greedy), 8 prompts (code, an explanation,
arithmetic, an agent's debugging step, a short story, a list, a bash one-liner, a derivative). The 4-bit file wrote
the answers (1,272 and 1,269 tokens); each low-bit file was then fed those answers token by token (teacher forcing)
and its top word and top-two gap were read at every position. A check of the method: the 4-bit file fed its own
answers agreed at every position (100%, both families).

| Low-bit file (size), against | Agrees unchecked | Flagged at gap < 2 | Agreement after | Flagged at gap < 3 | Agreement after |
| --- | ---: | ---: | ---: | ---: | ---: |
| bartowski IQ1_M (9.4 GB), against bartowski IQ4_XS | 83.7% | 35.7% | 97.8% | 45.6% | 99.8% |
| bartowski IQ2_XXS (10.7 GB), against bartowski IQ4_XS | 88.4% | 31.1% | 99.3% | 40.6% | 99.8% |
| Unsloth UD-IQ1_M (11.4 GB), against Unsloth UD-IQ4_XS | 89.8% | 28.1% | 99.5% | 37.4% | 100.0% |
| Unsloth UD-IQ2_XXS (11.8 GB), against Unsloth UD-IQ4_XS | 89.5% | 28.1% | 99.6% | 37.0% | 99.9% |

"Agreement after" assumes every flagged word takes the 4-bit model's word. The gap is a good signal: the
disagreements sit where the top two are close (gap under 0.5: 46-57% of those words disagree; gap 3-5: 0-1.8%; over 5,
about half of all words: none), and with Unsloth's UD-IQ1_M no word with a gap of 3 or more disagreed.

Unsloth's "1-bit" file keeps its important layers at higher precision: 11.4 GB against 11.8 GB for its own 2-bit one,
and it agrees as well. At this model size 1 bit saves almost no memory.

## Why it is rejected for models that fit

A flagged word costs a whole 4-bit step (every layer again: a changed layer changes the next one's input). With a
third of the words flagged and the cheap pass at half a 4-bit step, a word costs about 0.5 + 0.33 = 0.83 of a 4-bit
step: little gain, for a little less than the 4-bit model's answers. It pays only where a 4-bit step costs several
times a cheap one, for example when the 4-bit file is mostly on the SSD and the cheap one is in memory. Where a 3-bit
file fits and a 4-bit one does not, running the 3-bit file is the simpler choice and already works.

Not tried: a 4-bit GGUF's codes split into their top and bottom 2 bits (the cheap pass reads half the bytes, the exact
one both, stored once; Any-Precision LLM, 2024, is a published form of this nesting). A test copy with every weight's
bottom 2 bits set to the middle value would say how good the cheap half is; even a good result would only lower the
flagged share somewhat.

## Caveats

One model and family size, 8 prompts, greedy decoding (with sampling a different word is not a sign of an error).
Teacher forcing is the best case: in real use the cheap model's unflagged mistakes change what follows. Agreement is
with the 4-bit file, not with the full-precision model.

## If it is reopened

The measurement is about an hour for a new pair of files: `tools/confidence_gate.py ref` on the 4-bit file, `probe` on the
low-bit one (llama-server, `/completion` with `n_probs: 2`; the gap is the difference of the top two log-probabilities),
then the flagged share and the slips per gate. The model to try first: DeepSeek-V4 Flash (Unsloth UD-IQ1_S, 82.5 GB,
against UD-Q4_K_XL, 155 GB) on the Spark, if MoMoE runs it.
