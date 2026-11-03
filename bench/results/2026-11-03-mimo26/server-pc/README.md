# Community benchmark on NVIDIA GeForce RTX 4070 Ti

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti, 12 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.1.0-17-amd64; GPU driver: 580.178.04; CUDA: 13.0
- Momoe: project 0.1.55, commit d6686fb3; engine momoe-metal-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: mimo-v2.6-flash-mopd-bpw2.5
- Context: 32768 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
momoe-cuda -m MiMo-V2.6-Flash-MOPD-BPW2.5-00001-of-00003.gguf --ctx 32768 --chunk 1024
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
| short chat | 43 (40-48) | 0 | 100 (100-100) | 9 | not measured | 5.4 (4.9-6.5) | 9.01 (7.96-10.61) |
| 4K prompt, fresh | 4020 (4020-4020) | 0 (0-0) | 24 cap | 3 | 151.1 (142.1-161.1) | not measured | 26.71 (25.09-28.34) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 229.5 (226.6-239.0) | not measured | 70.17 (67.42-70.97) |
| agent session, follow-up turns | 14626 (14619-14674) at the last turn | see below | up to 64 | 3 | see below | 4.2 (3.3-4.5) | 79.49 (45.08-103.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.1 % (68.1-68.2); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 13968 (13968-13972) new tokens of the files are read once); first turn (nothing cached) first token: 21.21 s (20.38-21.96).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 421 | 0 | 421 | 19.5 | 21.96 | 3.7 |
| 1 | 3728 | 445 | 3283 | 32.7 | 100.673 | 4.0 |
| 2 | 5248 | 3782 | 1466 | 32.6 | 45.084 | 4.1 |
| 3 | 8673 | 5300 | 3373 | 32.7 | 103.126 | 4.0 |
| 4 | 11588 | 8722 | 2866 | 36.6 | 78.4 | 4.1 |
| 5 | 14619 | 11634 | 2985 | 38.5 | 77.523 | 3.3 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 2140 s.
