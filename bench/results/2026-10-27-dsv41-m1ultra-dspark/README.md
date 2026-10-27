# Community benchmark on Apple M1 Ultra

Measured on 2026-10-27 with `tools/strata_bench.py` (suite 1, 2 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Ultra, 48 GiB VRAM
- CPU: arm (20 cores, 20 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 24.6.0; GPU driver: not read; CUDA: not read
- Momoe: project 0.1.51, commit 67ff3a10; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: deepseek-v4.1-flash-ist-das-2.00bpw
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --runs 2` against the HTTP API of a Momoe server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 2 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 2 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 2 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 32 (29-38) | 0 | 100 (100-100) | 6 | not measured | 7.8 (6.8-8.2) | 3.66 (3.30-4.23) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 2 | 79.0 (77.9-80.2) | not measured | 52.20 (51.45-52.95) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 2 | 69.6 (69.3-69.9) | not measured | 238.09 (237.12-239.06) |
| agent session, follow-up turns | 14135 (14133-14137) at the last turn | see below | up to 64 | 2 | see below | 6.5 (6.0-6.9) | 97.49 (47.55-101.65) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.0 % (69.0-69.0); earlier-turn tokens read again per session: 1 (1-1) (earlier turns that the cache did not hold; the 13285 (13285-13285) new tokens of the files are read once); first turn (nothing cached) first token: 23.43 s (23.37-23.49).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 25.7 | 23.49 | 5.7 |
| 1 | 3767 | 640 | 3127 | 30.8 | 101.651 | 6.6 |
| 2 | 5279 | 3826 | 1453 | 30.5 | 47.612 | 6.5 |
| 3 | 8404 | 5342 | 3062 | 31.4 | 97.523 | 6.2 |
| 4 | 11248 | 8447 | 2801 | 29.6 | 94.638 | 6.7 |
| 5 | 14133 | 11290 | 2843 | 29.1 | 97.678 | 6.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1769 s.
