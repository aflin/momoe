# Community benchmark on Apple M1 Ultra

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Apple M1 Ultra, 48 GiB VRAM
- CPU: arm (20 cores, 20 threads); installed RAM: 64.0 GiB; storage: not measured
- OS: Darwin 24.6.0; GPU driver: not read; CUDA: not read
- Momoe: project 0.1.55, commit 04c4e306; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: mimo-v2.6-flash-mopd-bpw2.5
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-metal -m MiMo-V2.6-Flash-MOPD-BPW2.5-00001-of-00003.gguf --ctx 32768 --pool-gib 25 --nocache
```

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Momoe server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 43 (40-48) | 0 | 100 (100-100) | 9 | not measured | 7.0 (5.7-7.8) | 4.84 (4.61-5.30) |
| 4K prompt, fresh | 4020 (4020-4020) | 0 (0-0) | 24 cap | 3 | 165.5 (150.7-165.8) | not measured | 24.34 (24.31-26.68) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 202.2 (202.1-206.2) | not measured | 79.58 (77.92-79.58) |
| agent session, follow-up turns | 14684 (14679-14691) at the last turn | see below | up to 64 | 3 | see below | 5.6 (5.2-5.9) | 63.62 (31.96-75.05) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.2 % (68.2-68.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14000 (14000-14000) new tokens of the files are read once); first turn (nothing cached) first token: 10.75 s (10.48-10.76).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 421 | 0 | 421 | 40.6 | 10.483 | 4.5 |
| 1 | 3759 | 445 | 3314 | 45.9 | 72.173 | 5.2 |
| 2 | 5289 | 3822 | 1467 | 45.4 | 32.33 | 5.2 |
| 3 | 8725 | 5352 | 3373 | 44.9 | 75.05 | 5.3 |
| 4 | 11643 | 8777 | 2866 | 44.9 | 63.77 | 5.8 |
| 5 | 14679 | 11694 | 2985 | 47.1 | 63.422 | 5.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 1677 s.
