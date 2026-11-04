# Community benchmark on NVIDIA GeForce RTX 4070 Ti + NVIDIA GeForce RTX 3060

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti + NVIDIA GeForce RTX 3060, 24 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.1.0-17-amd64; GPU driver: 580.178.04; CUDA: 13.0
- Momoe: project 0.1.55, commit 5cbf2c5f; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: minimax-m3.1-flash-ud-iq2_xxs
- Context: 16384 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-cuda -m MiniMax-M3.1-Flash-UD-IQ2_XXS-00001-of-00004.gguf --ctx 16384 --chunk 512 --layer-split auto
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Momoe server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 219 (216-224) | 0 | 100 (100-100) | 9 | not measured | 2.8 (2.7-3.1) | 29.21 (28.24-30.70) |
| 4K prompt, fresh | 4100 (4100-4100) | 0 (0-0) | 24 cap | 3 | 22.7 (22.7-23.1) | not measured | 181.96 (177.71-182.79) |
| agent session, follow-up turns | 13929 (13915-13931) at the last turn | see below | up to 64 | 3 | see below | 2.7 (2.5-3.0) | 233.50 (113.22-242.02) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.4); earlier-turn tokens read again per session: 137 (137-137) (earlier turns that the cache did not hold; the 12901 (12889-12902) new tokens of the files are read once); first turn (nothing cached) first token: 77.44 s (75.09-77.83).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 762 | 0 | 762 | 10.5 | 75.095 | 2.4 |
| 1 | 3863 | 773 | 3090 | 12.8 | 241.355 | 2.6 |
| 2 | 5335 | 3860 | 1475 | 13.0 | 114.257 | 2.6 |
| 3 | 8385 | 5398 | 2987 | 12.7 | 235.382 | 2.9 |
| 4 | 11073 | 8382 | 2691 | 11.8 | 228.405 | 2.8 |
| 5 | 13915 | 11132 | 2783 | 11.9 | 234.961 | 2.8 |

### Skipped and failed

- skipped 16K: the context is 16384 tokens

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 5086 s.
