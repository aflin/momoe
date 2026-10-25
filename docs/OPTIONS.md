# MoMoE's options

Every option of the `momoe` command, of setup, of a model's config file and of MoMoE's own engines (`momoe-cuda`,
`momoe-metal`), in one place. Strata's engine (`engine/momoe`, the Flash-Next and Qwen3.6 families on a PC) has its
own list: see [the last section](#stratas-engine-enginemomoe).

Most people need only the first two sections: setup writes the config, and `momoe start` runs it.

## The `momoe` command

| Command | What it does |
| --- | --- |
| `momoe setup [setup options]` | install a model or change its settings, then start it ([setup's options](#setup-momoe-setup)) |
| `momoe start CONFIG [options]` | start the server from `config/CONFIG.json`, in the background (`.json` can be left off) |
| `momoe stop` | stop it |
| `momoe status` | whether it runs, which model, on which port |
| `momoe list` | the configs in `config/` |
| `momoe log [server\|engine\|unsloth\|lmstudio\|all] [-n N] [--no-follow]` | the last N lines (20) of the logs, followed as they grow; by default the server's and the engine's |
| `momoe run claude\|codex\|opencode [--config CONFIG] [their arguments]` | that agent, connected to this server (started first if needed) |
| `momoe run unsloth\|lmstudio [--config CONFIG]` | that app, pointed at this server |
| `momoe remove CONFIG\|FOLDER [--dry-run] [-y] [--wipe]` | delete a model: its GGUF, its pack, the config and its logs (also `momoe delete`); lists them and asks first |
| `momoe help` | the same list, with more words |

Options of `momoe start` for that start only (add `--save` to write them into the config):

| Option | What |
| --- | --- |
| `--host HOST` | where the server listens: `127.0.0.1` (default, this PC only) or `0.0.0.0` (other devices too: give `--api-key` with it) |
| `--api-key KEY` | clients must send this key (several: `key1,key2`) |
| `--draft-vocab cjk\|en\|cyrillic\|fr` | the draft layer's tokens (see setup's `--draft-vocab`) |
| `--vram-reserve-mib N` | VRAM the engine leaves free for other programs |
| `--save` | keep the four above in the config |
| `--gpu N`, `--port N` | the GPU (as `nvidia-smi` numbers them) and the port, for this start only |
| `--foreground` | run in this terminal instead of in the background |

`momoe run`: `--config` goes right after the tool's name; later arguments are the tool's own. `momoe remove`: `-y`
asks nothing, `--dry-run` only lists, `--wipe` also deletes the other configs that use the same files.

## Setup (`momoe setup`)

`momoe setup` asks its questions; `--yes` takes the recommended answers. An option given on the command line answers
its question. "Families" says which models use it: **Flash-Next** is the Flash-Next and Qwen3.6 families on Strata's
engine (on a PC), **ggml** is every model on `momoe-cuda` / `momoe-metal` (also the Flash-Next and Qwen3.6 families
on a Mac or the Spark). The others are used by both.

### The model

| Option | Families | What |
| --- | --- | --- |
| `--family NAME` | | `qwen` (Flash-Next), `swift`, `coder`, `unsloth` (Flash-Next's 4-bit files), `qwen36`, `qwen36-bartowski`, `ornith`, `kat-coder`, `bolt`, `glm53`, `ornith122`, `flint`, `lightning`, `gemma4`, `lfm25`, `lfm25-8b` |
| `--model SIZE` | | the file size (quantization), for example `IQ3_XXS`, `UD-IQ4_XS`, `UD-Q4_K_XL`; `./setup.sh --check` lists what fits |
| `--context N` | | the context in tokens (default: setup's choice for the memory) |
| `--vision yes\|no\|gpu\|cpu` | | pictures (off unless asked for): download the image encoder; `gpu` / `cpu`: where it runs |
| `--vision-tokens N` | | the most tokens a picture becomes (default 1,024 on a GPU, 300 on the CPU; the Qwen3.6 architecture's models 1,024, which they need to read text) |
| `--draft-vocab cjk\|en\|cyrillic\|fr` | | the draft layer's tokens: `cjk` (Flash-Next's default), `en` (English and code: less VRAM, the ggml families' default), `cyrillic`, `fr` |
| `--kv int8\|q4_0\|k8v4` | Flash-Next | the KV cache's precision above 8K context (default `int8`) |
| `--rope-scaling none\|linear\|yarn`, `--rope-scale F` | Flash-Next | a context past the trained 262,144 tokens |
| `--experimental-speed-projection on\|off\|GGUF` | Flash-Next | EXPERIMENTAL, off by default: the speed control vector ([DETAILS.md](DETAILS.md#experimental-speed-projection-experimental-off-by-default)) |

### GPUs

| Option | Families | What |
| --- | --- | --- |
| `--gpu N` | | one GPU, as `nvidia-smi` numbers them (asked when there are several) |
| `--gpus 0,1` or `all` | | several GPUs for one model, the main one first ([MULTI_GPU.md](MULTI_GPU.md)) |
| `--layer-split K[,K2]` | | with `--gpus`: the first layer of each later GPU (default: placed from each one's free VRAM) |
| `--helper-tier` | | two NVIDIA GPUs: the second holds a second tier of experts instead of layers of its own ([README](../README.md#two-cards)) |
| `--no-remote-expert-opt` | Flash-Next | with two or more GPUs: leave out `--remote-expert-opt`, which setup adds there |
| `--cuda 12\|13\|auto` | Flash-Next | the CUDA toolkit of the engine (12: older cards, [OLDER_GPUS.md](OLDER_GPUS.md)) |
| `--backend cuda\|hip\|sycl` | Flash-Next | NVIDIA (default), AMD ([AMD_HIP.md](AMD_HIP.md)), Intel Arc (experimental) |
| `--vram-reserve-mib N` | Flash-Next | VRAM left free for other programs (the engine's default: 700) |

### Memory

| Option | Families | What |
| --- | --- | --- |
| `--low-ram auto\|on\|off\|resident\|mmap` | Flash-Next | read the experts from one file instead of copying them all into RAM |
| `--resident-budget-gib N` | Flash-Next | UD-Q4_K_XL, UD-IQ4_XS: the GiB of experts kept in RAM |
| `--kv-streaming auto\|on\|off` | Flash-Next | from 64K context: the KV cache in RAM, only the attention's window in VRAM |
| `--parallel N` | Flash-Next | up to N requests decode together (opt-in; each takes VRAM from the expert cache) |
| `--replicas N` | Flash-Next | several GPUs: N independent engines, requests spread over them |
| `--calibrate` | Flash-Next | tune the engine's settings for this PC (15-30 minutes), then start |

### The server and the files

| Option | What |
| --- | --- |
| `--port N` | the server's port (default 8080) |
| `--host HOST`, `--api-key KEY` | as for `momoe start` |
| `--config-name NAME` | the config's file name (default: asked) |
| `--data-dir DIR` | where models, packs and draft layers go (remembered for every MoMoE folder on this PC) |
| `--models-dir DIR` | where the GGUF files go |
| `--gguf-dir DIR` | use GGUF files you already have (a folder with every shard) |
| `--source auto\|huggingface\|modelscope` | where the files come from |
| `--no-browser` / `--browser` | open the chat page when the model is ready, or not (remembered) |
| `--yes` | the recommended answers, no questions |
| `--no-start` | install only |
| `--update` | update the engines, Python packages and settings without starting (what `update.sh` runs) |
| `--check` | only check this PC: what fits |
| `--inspect SOURCE [VARIANT]` | what a GGUF is and whether MoMoE runs it, from its headers (a file, folder, URL, `hf:owner/repo`, `ms:owner/repo`) |

`./setup.sh --help` prints the same list with longer explanations.

## The config file (`config/NAME.json`)

Setup writes the config; you can edit it. Setup rewrites only its own keys (the first table) when it runs again, and
keeps any other key you added.

### The keys setup writes

| Key | What |
| --- | --- |
| `exe` | the engine program: `engine/momoe`, `build-cuda/momoe-cuda` or `build-metal/momoe-metal` |
| `args` | the engine's arguments ([`momoe-cuda` / `momoe-metal`](#momoe-cuda-and-momoe-metal); Strata's engine: `engine/momoe --help`) |
| `cwd`, `log`, `lib_dirs` | the engine's working folder, its log file, the folders with its CUDA libraries |
| `tokenizer`, `model_name` | the tokenizer's folder and the name the API shows |
| `port`, `host`, `api_key` | where the server listens and the key it asks for |
| `gpu` | a GPU number, or a list: `[main, second]` (a layer split, or the helper tier with `--helper-tier` in `args`) |
| `gpus_asked`, `setup_tag` | setup's own notes: the GPU question was answered; the model the config was made for |
| `layer_split` | the layer split's first layers per later GPU, or `auto` |
| `vision` | the image encoder: `mmproj`, `model`, `gpu`, `min_tokens`, `max_tokens` |
| `draft_vocab` | the draft layer's tokens |
| `backend`, `env` | the GPU backend; environment variables for the engine (`{"STRATA_ROUTE_TAIL_SKIP": "0"}`) |
| `sampling` | the defaults for requests that give none: `temperature`, `top_p`, `top_k`, ... |
| `fit_max_tokens` | `false`: answer 400 when prompt plus `max_tokens` does not fit, as llama.cpp does (default: shorten `max_tokens`) |

### Keys you can add

All are off or at their default unless set.

| Key | What |
| --- | --- |
| `reasoning_effort` | the thinking level for requests that give none |
| `reasoning_budget_tokens` | the most thinking tokens a reply takes |
| `preserve_thinking` | keep earlier turns' thinking in the prompt (`true` / `false`; default: the chat template's) |
| `anthropic_thinking` | `"on_request"`: Anthropic requests think only when they ask to |
| `parallel`, `replicas` | as setup's `--parallel` and `--replicas` (Strata's engine) |
| `idle_unload_s` | unload the model after this many seconds without requests; the next request loads it again |
| `min_free_vram_mib` | load an unloaded model only when this much VRAM is free (else 503) |
| `before_load` (+ `_timeout_s`, `_strict`, `_ignore_failure`) | a command run before the model loads again (to unload another server's model) |
| `lazy_load` | start the API without the model; the first request loads it |
| `vram_elastic`, `vram_segment_mib` | give VRAM back to other programs while idle ([DETAILS.md](DETAILS.md#sharing-the-gpu-with-other-programs-optional)) |
| `prevent_sleep` | keep the PC awake while a request runs |
| `slot_save_path` | `POST /slots/0?action=save\|restore` (llama-server's API) to and from this folder |
| `mcp_servers` | MCP servers whose tools the web app's chat can use ([DETAILS.md](DETAILS.md#tools-from-mcp-servers)) |
| `aliases` | other model names the API accepts |
| `allowed_hosts`, `cors_origins`, `trusted_origins` | which hosts and web pages may use the server |
| `api_monitor` | the last 100 requests at `/api/requests` (kept in memory) |
| `engine_silence_s` | how long the engine may stay silent before the server ends it (0: forever) |
| `repeat_stop_tokens` | end a reply that repeats one token this many times (0: off) |
| `tool_call_recovery`, `reasoning_loop_recovery`, `reasoning_close_retry` | recover from a malformed tool call, a thinking loop, a reply that ends inside its thinking |
| `expert_profile_save` (+ `_every`) | save what the engine's expert cache learned to this file (Strata's engine) |
| `split_skip_if_fits`, `gpu_order` | a layer split runs on the first card when everything fits there; `"as_given"` keeps the config's GPU order |
| `threads`, `cuda`, `arch` | the engine's CPU threads; the CUDA 12 engine; the model family when the config names no model |
| `remote_expert_opt` | `false`: as setup's `--no-remote-expert-opt` |
| `images` | image generation (Qwen-Image), [IMAGES.md](IMAGES.md) |
| `open_browser` | `false`: as setup's `--no-browser` |
| `answer_room`, `effort_position`, `codex_thread_titles`, `codex_compaction_cache`, `env_to_user` | finer points of the chat template and of Codex; see [DETAILS.md](DETAILS.md) |

## `momoe-cuda` and `momoe-metal`

The same program (`src/metal/momoe_metal.cpp`) built for CUDA or for Metal. Setup writes its arguments into the
config's `args`; the server starts it with `--serve` and adds `--layer-split` / the GPU for a config's `"gpu"`.

### Arguments

| Argument | Engines | What |
| --- | --- | --- |
| `-m FILE.gguf` | both | the model (its first shard) |
| `--ctx N` | both | the context in tokens (default 32768) |
| `--chunk N` (or `--batch N`) | both | the prompt's chunk size in tokens (default 512; setup writes 1024 on CUDA) |
| `--pool-gib G` | both | the expert pool's size; the experts outside it are read from the SSD when a token needs them (default: fitted to the free memory) |
| `--slots N` | both | the pool as a number of experts instead of GiB |
| `--preload` | both | every expert of the pool loaded at the start (a pool that holds every expert does this by itself) |
| `--nocache` | Metal | expert reads bypass macOS's file cache, as on a Mac without the RAM to cache the file |
| `--direct-io` | CUDA | experts read from the SSD past the page cache (on by itself on a unified-memory machine of 16 GiB or more: the Spark) |
| `--mtp K` | both | draft up to K tokens with the model's MTP layer |
| `--mtp-gguf FILE` | both | the MTP layer from another file (Flash-Next: its GGUFs have none; setup packs one) |
| `--draft-vocab FILE` | both | the draft head's token subset (`data/draft_vocab_*.bin`) |
| `--spec-min-p P` | both | draft only while the MTP layer is this sure (default 0.5) |
| `--device N` | CUDA | the GPU, in CUDA's numbering (the server sets it from the config's `"gpu"`) |
| `--layer-split auto\|K[,K2]` | CUDA | layers in ranges, one per card (the server adds it for `"gpu": [a, b]`) |
| `--helper-tier` | CUDA | the second card holds a second pool under the main one's |
| `--helper-gib G` | CUDA | the helper pool's largest size (default: the helper's free VRAM) |
| `--helper-reserve-mib N` | CUDA | VRAM left free on the helper (default 512) |
| `--profile FILE` | both | pin the hottest experts of a Strata expert profile (`MOMOE_PIN_FRAC` of the pool, 0.6); measured slower |
| `--nostall sub\|drop` | both | a token's graph runs without pauses, choosing among the experts in the pool (changes the output; [src/metal/README.md](../src/metal/README.md#no-stall-mode---nostall-subdrop-opt-in-changes-the-output-2026-10-13)) |
| `--nostall-every N` | both | with `--nostall`: every Nth layer keeps its pause |
| `--serve` | both | the server's protocol on stdin/stdout (what `momoe start` uses) |
| `--vision` | both | accepted for the server; pictures always work with `--serve` |
| `--tokens 1,2,3`, `--tokens-file FILE`, `-n N` | both | without `--serve`: a prompt as token ids, and the tokens to write (default 32) |
| `--bench-window` | both | a test: the cost of a verify window of T tokens |

### Environment variables: settings

Set them in the config's `env`, for example `"env": {"MOMOE_READERS": "8"}`.

| Variable | Engines | What |
| --- | --- | --- |
| `MOMOE_READERS=N` | both | threads that read experts into the pool (default 16 on CUDA, 8 on Metal, 4 on a unified-memory machine under 16 GiB) |
| `MOMOE_DEMAND_READERS=N` | both | separate threads for the reads a layer waits for, never behind the predicted ones (default: none) |
| `MOMOE_CPU_THREADS=N` | CUDA | threads that compute the experts the GPU does not hold (default: half the hardware threads) |
| `MOMOE_FILLERS=N` | CUDA | background threads that fill the pool while the CPU computes experts (default 2) |
| `MOMOE_FILLS=N` | both | the most experts put out of the pool a token (default 16; Nemotron: no limit) |
| `MOMOE_ADMIT=N` | CUDA | how often an expert must be used lately before it is copied into the pool (default 2) |
| `MOMOE_POLICY=lru` | both | the pool keeps the most recently used experts (default: the most often used, measured faster) |
| `MOMOE_PIN_FRAC=F` | both | with `--profile`: the share of the pool pinned (default 0.6) |
| `MOMOE_VRAM_MARGIN_MIB=N` | CUDA | VRAM the pool leaves free (default 512) |
| `MOMOE_METAL_MARGIN_MIB=N` | Metal | the same for Metal's working set (default 512) |
| `MOMOE_UMA_RESERVE_GIB=G`, `MOMOE_UMA_MIN_GIB=G` | CUDA | unified memory (Spark, Jetson): memory kept for the system (10), and the least free memory to start (24); under 16 GiB 20% and 15% of it |
| `MOMOE_UMA_POOL_SHARE=F` | CUDA | a unified-memory machine under 16 GiB: the pool's largest share of the memory (default 0.3) |
| `MOMOE_DIRECT=1` | CUDA | as `--direct-io` |
| `MOMOE_NO_DIRECT=1` | both | CUDA: no `--direct-io` on a unified-memory machine; Metal: the router's choice reaches the experts through a copy, as before the direct choice |
| `MOMOE_NO_HYBRID=1` | CUDA | the CPU does not compute experts: every expert the GPU lacks is copied to it |
| `MOMOE_QWEN_HYBRID=0\|1`, `MOMOE_HYB_FRAC=F` | CUDA | the Qwen3.6 architecture: copies (0) or the CPU's experts (1) without the start-up trial; or the split with that share copied |
| `MOMOE_NO_MAILBOX=1` | CUDA | the CPU's experts through the scheduler's stops instead of mailboxes |
| `MOMOE_RMB=1`, `MOMOE_NO_RMB=1`, `MOMOE_POOL_HOST=1` | CUDA | unified memory: resident mailboxes on or off, the pool in host memory |
| `MOMOE_COPY_KERNEL=0` | CUDA | resident mailboxes copy with `cudaMemcpyAsync` instead of a kernel |
| `MOMOE_CUDA_GRAPHS=1` | CUDA | CUDA graphs on where the engine turns them off (`GGML_CUDA_DISABLE_GRAPHS=1`: off everywhere) |
| `MOMOE_EMBD_GPU=1` | CUDA | the token embeddings in VRAM (default: in RAM, as llama.cpp keeps them) |
| `MOMOE_NO_PREDICT=1` | both | do not read the next layer's predicted experts ahead |
| `MOMOE_PREDICT2=1` | both | also predict two layers ahead (as fast, a third more read) |
| `MOMOE_PREDICT_K=N` | both | read only each token's N most likely predicted experts |
| `MOMOE_NO_PRELOAD=1` | both | a pool that holds every expert is not preloaded |
| `MOMOE_NO_PAUSE=1`, `MOMOE_FORCE_PAUSE=1` | Metal | the scheduler's callback instead of pause points; pauses even after a preload |
| `MOMOE_NO_GRAPH_REUSE=1` | both | build each token's graph again instead of keeping it |
| `MOMOE_NO_METAL_FIT=1` | Metal | do not shrink the pool to Metal's working set |
| `MOMOE_STAGE_NO_POOL=1` | CUDA | an expert the pool holds is read again instead of copied inside the GPU |

### Environment variables: tests and debugging

These change the output or only print; not for normal use.

| Variable | What |
| --- | --- |
| `MOMOE_DEBUG=1`, `MOMOE_DEBUG_LAST=1` | tensor sums, comparable with llama.cpp's `llama-eval-callback` (the last position only) |
| `MOMOE_TOP=1` | the 5 most likely tokens at each step |
| `MOMOE_KL_DUMP=FILE`, `MOMOE_KL_REF=FILE` | write a run's tokens and log-probabilities; feed them to another run and report KL per step |
| `MOMOE_TF_WINDOW=N`, `MOMOE_TF_FILE=FILE`, `MOMOE_TF_PROBS=FILE` | feed the tokens in a file in windows of N and compare the next tokens (top 5 probabilities to a file) |
| `MOMOE_BENCH_ALL=1`, `MOMOE_BENCH_SNAP=1` | with `--bench-window`: every position's output; the state saved as for a verify window |
| `MOMOE_FORCE_CB=1` | the scheduler's callback path, no kept graphs |
| `MOMOE_NO_QSA=1`, `MOMOE_FLINT_NO_RING=1`, `MOMOE_FLINT_NO_SCALE=1` | Flash-Next's attention to every position; Flint without its sliding-window ring, or without its down-projection scale |
| `MOMOE_TEST_SKIP_CPU=1` | skip the CPU's experts (timing only: the output is wrong) |
| `MOMOE_CPU_PROF=1`, `MOMOE_SEGTIME=1` | timings of the CPU's experts; GPU time between stops |
| `MOMOE_MB_DEBUG=1`, `MOMOE_RMB_DEBUG=1`, `MOMOE_METAL_FIT_DEBUG=1`, `MOMOE_GGML_DEBUG=1` | the mailboxes', the Metal fit's and ggml's debug lines |

## Strata's engine (`engine/momoe`)

It runs the Flash-Next and Qwen3.6 families on a PC. It has many more settings, inherited from Strata:

- **Its arguments:** `engine/momoe --help` lists them all (about 270 lines). Setup writes the ones a config needs.
- **Its environment variables** (`STRATA_*`, set in the config's `env`): the measured ones are in
  [DETAILS.md](DETAILS.md), each in its own section, for example
  [the tail skip](DETAILS.md#missed-experts-in-a-verify-window-the-tail-is-skipped-on-by-default-on-cuda-strata_route_tail_skip),
  [the CPU share of short prompts](DETAILS.md#short-prompts-the-cpu-shares-the-experts-on-by-default-on-one-nvidia-gpu-strata_prefill_cpu_share)
  and [the opt-in switches](DETAILS.md#more-opt-in-switches-measured-for-0141-all-off-unless-you-set-them).
- **Two GPUs:** [MULTI_GPU.md](MULTI_GPU.md), [SECOND_GPU.md](SECOND_GPU.md) and the README's
  [Two cards](../README.md#two-cards).
