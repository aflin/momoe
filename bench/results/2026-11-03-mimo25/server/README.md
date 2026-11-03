# Community benchmark on NVIDIA GeForce RTX 4070 Ti

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti, 12 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.1.0-17-amd64; GPU driver: 580.178.04; CUDA: 13.0
- Momoe: project 0.1.55, commit 7304c605; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: mimo-v2.5-ud-iq2_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Momoe server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 60 (57-65) | 0 | 100 (100-100) | 9 | not measured | 5.9 (5.3-6.2) | 10.54 (10.03-11.90) |
| 4K prompt, fresh | 4020 (4020-4020) | 0 (0-0) | 24 cap | 3 | 127.9 (125.6-136.9) | not measured | 31.55 (29.47-32.05) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 203.5 (201.2-215.2) | not measured | 79.15 (74.73-80.08) |
| agent session, follow-up turns | 14913 (14900-14936) at the last turn | see below | up to 64 | 3 | see below | 4.5 (4.0-4.9) | 74.47 (42.91-100.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.0-69.1); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14012 (14012-14013) new tokens of the files are read once); first turn (nothing cached) first token: 22.38 s (22.25-24.25).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 658 | 0 | 658 | 27.5 | 24.249 | 3.7 |
| 1 | 4012 | 685 | 3327 | 35.4 | 94.11 | 4.0 |
| 2 | 5535 | 4069 | 1466 | 33.8 | 43.426 | 4.3 |
| 3 | 8961 | 5588 | 3373 | 36.4 | 92.837 | 4.7 |
| 4 | 11877 | 9011 | 2866 | 39.0 | 73.575 | 4.5 |
| 5 | 14913 | 11928 | 2985 | 40.1 | 74.466 | 4.4 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2098 s.
