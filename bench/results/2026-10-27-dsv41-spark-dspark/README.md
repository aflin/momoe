# Community benchmark on NVIDIA GB10

Measured on 2026-10-27 with `tools/strata_bench.py` (suite 1, 2 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.7 GiB; storage: not measured
- OS: Linux 6.17.0-1026-nvidia; GPU driver: 580.159.03; CUDA: 13.0
- Momoe: project 0.1.51, commit eae00a50; engine momoe-metal-0.1
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 6 | not measured | 19.2 (14.9-33.4) | 0.78 (0.57-0.98) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 2 | 268.4 (175.2-361.7) | not measured | 17.50 (11.43-23.56) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 2 | 242.4 (165.1-319.8) | not measured | 76.12 (51.86-100.38) |
| agent session, follow-up turns | 14128 (14124-14132) at the last turn | see below | up to 64 | 2 | see below | 19.6 (16.1-23.2) | 22.11 (11.10-44.80) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.0 % (69.0-69.0); earlier-turn tokens read again per session: 1 (1-1) (earlier turns that the cache did not hold; the 13282 (13282-13282) new tokens of the files are read once); first turn (nothing cached) first token: 8.03 s (5.52-10.54).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 57.3 | 10.537 | 15.3 |
| 1 | 3764 | 640 | 3124 | 69.8 | 44.797 | 16.1 |
| 2 | 5271 | 3818 | 1453 | 64.7 | 22.476 | 20.3 |
| 3 | 8396 | 5334 | 3062 | 71.0 | 43.174 | 18.2 |
| 4 | 11244 | 8443 | 2801 | 69.6 | 40.293 | 20.4 |
| 5 | 14124 | 11281 | 2843 | 69.4 | 40.974 | 19.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 577 s.
