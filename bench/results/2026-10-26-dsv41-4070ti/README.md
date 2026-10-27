# Community benchmark on NVIDIA GeForce RTX 4070 Ti

Measured on 2026-10-26 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: NVIDIA GeForce RTX 4070 Ti, 12 GiB VRAM
- CPU: AMD Ryzen 9 5950X 16-Core Processor (16 cores, 32 threads); installed RAM: 62.7 GiB; storage: not measured
- OS: Linux 6.1.0-17-amd64; GPU driver: 580.178.04; CUDA: 13.0
- Momoe: project 0.1.51, commit 24d4c669; engine ds4-0.1
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: deepseek-v4.1-flash-ist-das-2.00bpw
- Context: 8192 tokens; parallel slots: 1
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
| short chat | 32 (29-38) | 0 | 100 (100-100) | 9 | not measured | 3.8 (3.6-4.8) | 13.42 (9.11-15.19) |
| 4K prompt, fresh | 4125 (4125-4125) | 0 (0-0) | 24 cap | 3 | 81.7 (49.4-102.8) | not measured | 50.47 (40.12-83.57) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Skipped and failed

- skipped 16K: the context is 8192 tokens
- skipped agent: context 8192 is below ~10K

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 648 s.
