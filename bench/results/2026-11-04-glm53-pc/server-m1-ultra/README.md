# Community benchmark on Apple M1 Ultra

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Ultra, 48 GiB VRAM
- CPU: arm (20 cores, 20 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 24.6.0; GPU driver: not read; CUDA: not read
- Momoe: project 0.1.55, commit 0a8b74d6; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: glm-5.3-flash-ud-iq2_xxs
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-metal -m GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf --ctx 65536 --pool-gib 26 --nocache
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
| short chat | 42 (39-47) | 0 | 100 (100-100) | 9 | not measured | 5.3 (4.2-5.8) | 5.41 (5.20-6.13) |
| 4K prompt, fresh | 4021 (4021-4021) | 0 (0-0) | 24 cap | 3 | 86.0 (81.1-87.2) | not measured | 46.81 (46.17-49.59) |
| 16K prompt, fresh | 16057 (16057-16057) | 0 (0-0) | 24 cap | 3 | 94.9 (92.5-95.1) | not measured | 169.35 (168.91-173.57) |
| agent session, follow-up turns | 13796 (13748-13875) at the last turn | see below | up to 64 | 3 | see below | 4.3 (3.8-4.6) | 66.36 (32.55-71.60) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.2 % (68.1-68.4); earlier-turn tokens read again per session: 200 (135-330) (earlier turns that the cache did not hold; the 13022 (12933-13084) new tokens of the files are read once); first turn (nothing cached) first token: 11.15 s (10.88-11.19).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 495 | 0 | 495 | 45.9 | 10.881 | 3.9 |
| 1 | 3602 | 493 | 3109 | 44.8 | 69.38 | 4.1 |
| 2 | 5083 | 3600 | 1483 | 45.6 | 32.547 | 4.1 |
| 3 | 8162 | 5081 | 3081 | 46.0 | 67.011 | 4.0 |
| 4 | 10938 | 8160 | 2778 | 43.0 | 64.589 | 3.8 |
| 5 | 13748 | 10936 | 2812 | 43.2 | 65.054 | 4.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2165 s.
