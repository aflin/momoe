# Community benchmark on Apple M1 Ultra

Measured on 2026-11-04 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Ultra, 48 GiB VRAM
- CPU: arm (20 cores, 20 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 24.6.0; GPU driver: not read; CUDA: not read
- Momoe: project 0.1.55, commit f199ee4d; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: minimax-m3.1-flash-ud-iq2_xxs
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-metal -m MiniMax-M3.1-Flash-UD-IQ2_XXS-00001-of-00004.gguf --ctx 32768 --pool-gib 20 --nocache
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
| short chat | 219 (216-224) | 0 | 100 (100-100) | 9 | not measured | 2.7 (2.3-3.0) | 14.37 (13.99-14.68) |
| 4K prompt, fresh | 4100 (4100-4100) | 0 (0-0) | 24 cap | 3 | 58.1 (55.5-58.3) | not measured | 71.05 (70.88-74.29) |
| 16K prompt, fresh | 16388 (16388-16388) | 0 (0-0) | 24 cap | 3 | 69.2 (67.0-69.8) | not measured | 237.86 (235.86-245.11) |
| agent session, follow-up turns | 13915 (13913-13939) at the last turn | see below | up to 64 | 3 | see below | 2.6 (2.1-3.0) | 99.00 (49.51-106.25) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.5 % (69.4-69.5); earlier-turn tokens read again per session: 71 (71-137) (earlier turns that the cache did not hold; the 12907 (12884-12909) new tokens of the files are read once); first turn (nothing cached) first token: 32.27 s (32.21-32.29).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 762 | 0 | 762 | 24.3 | 32.287 | 2.5 |
| 1 | 3854 | 773 | 3081 | 31.2 | 99.17 | 2.1 |
| 2 | 5329 | 3851 | 1478 | 30.1 | 49.51 | 2.1 |
| 3 | 8379 | 5392 | 2987 | 30.3 | 99.001 | 2.3 |
| 4 | 11068 | 8376 | 2692 | 27.8 | 97.324 | 2.5 |
| 5 | 13913 | 11130 | 2783 | 28.4 | 98.501 | 2.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 3374 s.
