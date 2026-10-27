# Community benchmark on Apple M1 Ultra

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 2 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Ultra, 48 GiB VRAM
- CPU: arm (20 cores, 20 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 24.6.0; GPU driver: not read; CUDA: not read
- Momoe: project 0.1.51, commit e83d3fc2; engine momoe-metal-0.1
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 6 | not measured | 7.4 (6.7-8.4) | 3.28 (2.82-3.71) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 2 | 82.6 (80.4-84.8) | not measured | 50.00 (48.67-51.32) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 2 | 71.8 (71.4-72.3) | not measured | 230.63 (229.20-232.07) |
| agent session, follow-up turns | 14091 (14091-14091) at the last turn | see below | up to 64 | 2 | see below | 5.9 (5.2-6.3) | 89.84 (44.30-91.39) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-68.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13225 (13225-13225) new tokens of the files are read once); first turn (nothing cached) first token: 22.01 s (21.78-22.24).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 27.1 | 22.237 | 6.3 |
| 1 | 3707 | 639 | 3068 | 33.9 | 90.385 | 6.0 |
| 2 | 5220 | 3766 | 1454 | 32.5 | 44.726 | 6.0 |
| 3 | 8345 | 5283 | 3062 | 33.5 | 91.394 | 5.9 |
| 4 | 11199 | 8397 | 2802 | 31.7 | 88.31 | 5.7 |
| 5 | 14091 | 11247 | 2844 | 31.2 | 91.232 | 5.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1682 s.
