# Community benchmark on NVIDIA GeForce RTX 4070 Ti + NVIDIA GeForce RTX 3060

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti + NVIDIA GeForce RTX 3060, 24 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.1.0-17-amd64; GPU driver: 580.178.04; CUDA: 13.0
- Momoe: project 0.1.55, commit 9534cb26; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: glm-5.3-flash-ud-iq2_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-cuda -m GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf --ctx 65536 --chunk 1024 --layer-split auto
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
| short chat | 41 (38-46) | 0 | 100 (100-100) | 9 | not measured | 4.7 (4.4-5.4) | 7.51 (6.84-9.87) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 82.1 (77.7-85.3) | not measured | 49.34 (47.22-51.81) |
| 16K prompt, fresh | 16057 (16057-16057) | 0 (0-0) | 24 cap | 3 | 122.0 (118.5-125.7) | not measured | 132.51 (128.53-135.67) |
| agent session, follow-up turns | 13806 (13726-13824) at the last turn | see below | up to 64 | 3 | see below | 4.0 (3.4-4.3) | 84.43 (50.60-97.73) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.2); earlier-turn tokens read again per session: 261 (200-330) (earlier turns that the cache did not hold; the 13024 (12912-13037) new tokens of the files are read once); first turn (nothing cached) first token: 23.46 s (22.53-24.09).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 494 | 0 | 494 | 21.7 | 23.462 | 4.1 |
| 1 | 3611 | 492 | 3119 | 32.0 | 97.731 | 4.1 |
| 2 | 5091 | 3609 | 1482 | 29.4 | 50.613 | 4.0 |
| 3 | 8170 | 5089 | 3081 | 36.0 | 85.796 | 4.0 |
| 4 | 10916 | 8168 | 2748 | 33.1 | 83.216 | 4.2 |
| 5 | 13726 | 10914 | 2812 | 33.3 | 84.434 | 3.7 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2461 s.
