# Community benchmark on NVIDIA GB10

Measured on 2026-10-27 with `tools/strata_bench.py` (suite 1, 2 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GB10, VRAM not read
- CPU: aarch64 (20 cores, 20 threads); installed RAM: 121.7 GiB; storage: not measured
- OS: Linux 6.17.0-1026-nvidia; GPU driver: 580.159.03; CUDA: 13.0
- Momoe: project 0.1.51, commit 54c5f7ca; engine momoe-metal-0.1
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 6 | not measured | 16.0 (14.2-18.4) | 0.79 (0.49-1.09) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 2 | 255.1 (143.7-366.4) | not measured | 19.99 (11.28-28.71) |
| 16K prompt, fresh | 16562 (16562-16562) | 0 (0-0) | 24 cap | 2 | 238.1 (155.0-321.2) | not measured | 79.26 (51.60-106.92) |
| agent session, follow-up turns | 14109 (14102-14116) at the last turn | see below | up to 64 | 2 | see below | 14.3 (13.2-17.3) | 21.71 (10.49-42.47) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-68.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13226 (13226-13226) new tokens of the files are read once); first turn (nothing cached) first token: 8.40 s (5.28-11.52).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 602 | 0 | 602 | 52.4 | 11.518 | 13.7 |
| 1 | 3707 | 639 | 3068 | 72.3 | 42.468 | 14.3 |
| 2 | 5225 | 3770 | 1455 | 65.6 | 22.205 | 14.4 |
| 3 | 8350 | 5288 | 3062 | 73.0 | 41.96 | 13.4 |
| 4 | 11203 | 8401 | 2802 | 68.7 | 40.806 | 13.5 |
| 5 | 14102 | 11258 | 2844 | 74.7 | 38.11 | 13.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 608 s.
