# MoMoE's options

Every option of the `momoe` command, of setup, of a model's config file and of MoMoE's own engines (`momoe-cuda`,
`momoe-metal`), in one place. Strata's engine (`engine/momoe`, the Flash-Next and Qwen3.6 families on a PC) has its
own list: see [the last section](#stratas-engine-enginemomoe).

Most people need only the first two sections: setup writes the config, and `momoe start` runs it.

Two engines, two kinds of setting: **Strata's engine** runs the Flash-Next and Qwen3.6 families on a PC with an NVIDIA
or AMD card; **`momoe-cuda` / `momoe-metal`** run every other model, and every model on a Mac, the DGX Spark and the
Jetson. Where an option is for one of them only, the tables say so.

## The `momoe` command

| Command | What it does |
| --- | --- |
| `momoe setup [setup options]` | install a model or change its settings, then start it ([setup's options](#setup-momoe-setup)); a running server is stopped first |
| `momoe start CONFIG [options]` | start the server from `config/CONFIG.json`, in the background (`.json` and `strata-` can be left off, or give a path) |
| `momoe stop` | stop it |
| `momoe status` | whether it runs, which model, on which port |
| `momoe list` | the configs in `config/` |
| `momoe log [server\|engine\|unsloth\|lmstudio\|all] [-n N] [--no-follow]` | the last N lines (20 by default; also `--lines N`) of the logs, followed as they grow. Without a name: the server's log and the engine log of the running config (or the last one started) |
| `momoe run claude\|codex\|opencode [--config CONFIG] [their arguments]` | that agent, connected to this server (started first if needed) |
| `momoe run unsloth\|lmstudio [--config CONFIG]` | that app, pointed at this server |
| `momoe remove CONFIG\|FOLDER [--dry-run] [-y] [--wipe]` | delete a model: its GGUF, its pack, the config and its logs (also `momoe delete`); lists them and asks first |
| `momoe help` | the same list, with more words |

`momoe run`: `--config` goes right after the tool's name (or before it); later arguments are the tool's own. With a
server running, a `--config` naming another config is refused. `momoe remove`: `-y` (or `--yes`) asks nothing, but
files other configs use stay; `--wipe` deletes those configs and files too; `--dry-run` only lists.

### Options of `momoe start`

| Option | Kept in the config | What |
| --- | --- | --- |
| `--host HOST` | with `--save` | where the server listens: `127.0.0.1` (default, this PC only) or `0.0.0.0` (other devices too: give `--api-key` with it) |
| `--api-key KEY` | with `--save` | clients must send this key (several: `key1,key2`); `--api-key "" --save` removes it |
| `--draft-vocab cjk\|en\|cyrillic\|fr` | with `--save` | the draft layer's tokens (see setup's `--draft-vocab`) |
| `--vram-reserve-mib N` | with `--save` | Strata's engine: VRAM left free for other programs. Refused for `momoe-cuda` / `momoe-metal`: set `MOMOE_VRAM_MARGIN_MIB` / `MOMOE_METAL_MARGIN_MIB` in the config's `env` instead |
| `--save` | | keep the four above in the config |
| `--gpus 0,1` or `all`, `--layer-split K` | always | several GPUs for this model, as setup's (`--layer-split` only together with `--gpus`) |
| `--gpu N` | no | one GPU, as `nvidia-smi` numbers them, for this start |
| `--port N` | no | the port, for this start |
| `--yes` | | no questions |
| `--foreground` | | run in this terminal instead of in the background |

Other setup options given to `momoe start` are ignored: change them with `momoe setup`.

## Setup (`momoe setup`)

`momoe setup` asks its questions; `--yes` takes the recommended answers. An option given on the command line answers
its question. "Engine" says which engine's setup uses it: **Strata** (Strata's engine) or **ggml** (`momoe-cuda` /
`momoe-metal`); blank: both. Setup takes the ggml path for a ggml family, on a Mac, and on a GPU that shares the RAM
(the Spark, a Jetson).

### The model

| Option | Engine | What |
| --- | --- | --- |
| `--family NAME` | | `qwen` (Flash-Next), `swift`, `coder`, `unsloth` (Flash-Next's 4-bit files), `qwen36`, `qwen36-bartowski`, `ornith`, `kat-coder`, `bolt`, `glm53`, `ornith122`, `flint`, `lightning`, `gemma4`, `lfm25`, `lfm25-8b` |
| `--model SIZE` | | the file size (quantization), for example `IQ3_XXS`, `UD-IQ4_XS`, `UD-Q4_K_XL` (`./setup.sh --help` lists them all) |
| `--context N` | | the context in tokens (default: setup's choice for this machine) |
| `--vision yes\|no\|gpu\|cpu` | | pictures (off unless asked for; `none` is `no`): download the image encoder. `yes`: setup picks where it runs; `gpu` / `cpu` choose that on Strata's engine. On the ggml path it runs on the GPU on a Mac and the Spark, else on the CPU |
| `--vision-tokens N` | | the most tokens a picture becomes, 1 or more. Default: Strata's engine 1,024 with the encoder on the GPU, 300 on the CPU; the ggml path 300. The Qwen3.6 architecture at least 1,024, which it needs to read text; Gemma 4 at most 512 (a picture is read in one batch) |
| `--draft-vocab cjk\|en\|cyrillic\|fr` | | the tokens the draft layer can propose: `cjk` (with Chinese, Japanese and Korean; Strata's engine's default), `en` (English and code: less VRAM; the default of the Qwen3.6 architecture on the ggml path, the only ggml models that use it), `cyrillic`, `fr` |
| `--kv int8\|q4_0\|k8v4` | Strata | the KV cache's precision above 8K context (default `int8`) |
| `--rope-scaling none\|linear\|yarn`, `--rope-scale F` | Strata | a context past the trained 262,144 tokens |
| `--experimental-speed-projection on\|off\|GGUF` | Strata | EXPERIMENTAL, off by default: the speed control vector ([DETAILS.md](DETAILS.md#experimental-speed-projection-experimental-off-by-default)) |

### GPUs

| Option | Engine | What |
| --- | --- | --- |
| `--gpu N` | | one GPU, as `nvidia-smi` numbers them (AMD: as setup lists them). Default: Strata's engine asks when there are several; the ggml path takes the card with the most memory |
| `--gpus 0,1` or `all` | | several GPUs for one model, the main one first ([MULTI_GPU.md](MULTI_GPU.md)) |
| `--layer-split K[,K2]` | | with `--gpus`: the first layer of each later GPU, rising (default: placed from each card's free VRAM) |
| `--helper-tier` | | two NVIDIA GPUs: the second holds a second tier of experts instead of layers of its own ([README](../README.md#two-cards)). Strata's engine picks the two best cards; the ggml path needs them named: `--gpus MAIN,HELPER` |
| `--no-remote-expert-opt` | Strata | with two or more GPUs: leave out `--remote-expert-opt`, which setup adds there |
| `--cuda 12\|13\|auto` | Strata | the CUDA toolkit of the engine (default `auto`: 13; 12 for older cards, [OLDER_GPUS.md](OLDER_GPUS.md)) |
| `--backend cuda\|hip\|sycl` | Strata | NVIDIA (default), AMD ([AMD_HIP.md](AMD_HIP.md); chosen by itself when there is no usable NVIDIA card), Intel Arc (experimental, its own setup) |
| `--vram-reserve-mib N` | Strata | VRAM left free for other programs (the engine's default: 700) |

### Memory

| Option | Engine | What |
| --- | --- | --- |
| `--low-ram auto\|on\|off\|resident\|mmap` | Strata | read the experts from one file instead of copying them all into RAM (default `auto`: when they would not fit the RAM) |
| `--resident-budget-gib N` | Strata | UD-Q4_K_XL, UD-IQ4_XS: the GiB of experts kept in RAM |
| `--kv-streaming auto\|on\|off` | Strata | from 64K context: the KV cache in RAM, only the attention's window in VRAM (default `auto`) |
| `--parallel N` | Strata | up to N requests decode together (opt-in; each takes VRAM from the expert cache) |
| `--replicas N` | Strata | several GPUs: N independent engines, requests spread over them |

### The server and the files

| Option | Engine | What |
| --- | --- | --- |
| `--port N` | | the server's port (default 8080) |
| `--host HOST`, `--api-key KEY` | | as for `momoe start` |
| `--config-name NAME` | | the config's file name (default: asked; with `--yes`, the model's own name) |
| `--data-dir DIR` | | where models, packs and draft layers go (remembered for every MoMoE folder on this PC) |
| `--models-dir DIR` | | where the GGUF files go (default: the data folder's `models/`) |
| `--gguf-dir DIR` | | use GGUF files you already have (a folder with every shard): checked, never downloaded or deleted |
| `--source auto\|huggingface\|modelscope` | | where the files come from (default `auto`: Hugging Face) |
| `--yes` | | the recommended answers, no questions |
| `--no-start` | | install only |
| `--update` | | update the engines, Python packages and settings without starting (what `update.sh` runs) |
| `--check` | | only check this machine: what fits, nothing installed |
| `--inspect SOURCE [VARIANT]` | | what a GGUF is and whether MoMoE runs it, from its headers (a file, folder, URL, `hf:owner/repo`, `ms:owner/repo`) |

`./setup.sh --help` prints the same list with longer explanations.

## The config file (`config/NAME.json`)

Setup writes the config; you can edit it. When setup runs again it writes its own keys again (the first table) and
keeps the rest: your `sampling` block, your own `env` entries and `vision` keys, and every key of the second table
unless that setup run writes it too (`--parallel`, `--replicas`, `arch`, `cuda`). Engine options you added to `args`
are not kept (setup names the ones it drops). The earlier file is kept as `NAME.json.bak`.

**Strata's engine only:** the keys marked "Strata's engine" below add options for that engine. With `momoe-cuda` and
`momoe-metal` the server leaves them out, and its log says which.

### The keys setup writes

| Key | Values | What |
| --- | --- | --- |
| `exe` | a path | the engine program: `engine/momoe` (or `engine-cuda12/momoe`), `build-cuda/momoe-cuda` or `build-metal/momoe-metal` |
| `args` | a list of strings | the engine's arguments ([`momoe-cuda` / `momoe-metal`](#momoe-cuda-and-momoe-metal); Strata's engine: `engine/momoe --help`) |
| `cwd` | a folder | the engine's working folder; relative paths in the config are taken from it |
| `log` | a file | the engine's log (the image encoder writes to it too) |
| `lib_dirs` | a list of folders | put first on the library path: the engine's GPU libraries |
| `tokenizer` | a folder | the tokenizer |
| `model_name` | a string | the name the API shows (default: the family's) |
| `port` | a number; 8080 | the server's port |
| `host` | an address; `127.0.0.1` | where the server listens |
| `api_key` | a string (`key1,key2` for several) or a list | the key clients must send (empty: none) |
| `gpu` | a number, a list `[0, 1]` or `"0,1"` | the card(s), as `nvidia-smi` numbers them: `[main, second]` is a layer split, or the helper tier with `--helper-tier` in `args` |
| `gpus_asked` | `true` | setup's note: the two-GPU question was answered |
| `setup_tag` | a string | setup's note: the model the config was made for |
| `layer_split` | `"auto"` (default), or each later GPU's first layer: `"24,36"` or `[24, 36]` | where a layer split's cards start |
| `vision` | an object (below) | the image encoder |
| `draft_vocab` | `cjk`, `en`, `cyrillic`, `fr` | the draft layer's tokens (read by setup and `momoe start`, not by the server) |
| `backend` | `"hip"` (AMD) | without it: NVIDIA, or Metal on a Mac |
| `env` | an object of strings | environment variables for the engine and the image encoder, for example `{"MOMOE_READERS": "8"}` or Strata's `{"STRATA_ROUTE_TAIL_SKIP": "0"}` |
| `sampling` | an object (below) | the defaults for requests that give none; setup writes the model card's and keeps yours |
| `fit_max_tokens` | `true` (default) or `false` | `false`: answer 400 when prompt plus `max_tokens` does not fit, as llama.cpp does (default: shorten `max_tokens`). Setup writes `true` in `momoe-cuda` / `momoe-metal` configs |

**`vision`:** `exe` (the encoder program), `mmproj` (the encoder's file), `model` (the model's file), `gpu` (`true`:
on the GPU; default the CPU), `threads` (the encoder's CPU threads; setup writes half the cores), `max_tokens` (the
most tokens a picture becomes: default 1,024 on the GPU, 300 on the CPU, 0 no limit), `min_tokens` (the fewest;
setup writes 1,024 for the Qwen3.6 architecture). You can add `cuda_device` (the encoder on a card of its own) and,
on Strata's engine with `vram_elastic`, `elastic` (`true`: the encoder starts with the first picture), `elastic_mib`
(VRAM kept for it, 2048) and `elastic_idle_s` (stopped after this long without a picture, 120).

**`sampling`:** `temperature` (0 or more; 0: always the likeliest token), `top_p` (above 0, up to 1), `top_k` (1 to
64), `min_p` (0 to 1), `presence_penalty` and `frequency_penalty` (0 or more), `repetition_penalty` (above 0; also
`repeat_penalty`), `penalty_last_n` (0 or more; also `repeat_last_n`), `seed` (above 0), `experimental_speed_projection`
(`true` / `false`, Strata's engine). A bad value stops the start;
an unknown key is ignored with a note. Without the block: the model card's for the config's `arch`.

### Keys you can add

Each is at the default given here unless set.

| Key | Values | What |
| --- | --- | --- |
| `reasoning_effort` | `none`, `low`, `medium`, `high` | the thinking level for requests that give none |
| `reasoning_budget_tokens` | tokens; 0 (default): no limit | the most thinking tokens a reply takes; a request can set its own |
| `preserve_thinking` | `true` / `false`; default: the chat template's | keep earlier turns' thinking in the prompt |
| `anthropic_thinking` | `"model"` (default) or `"on_request"` | `"on_request"`: Anthropic requests think only when they ask to |
| `idle_unload_s` | seconds; 0 (default): never | unload the model after this long without requests; the next request loads it again |
| `min_free_vram_mib` | MiB; 0 (default): no check | load an unloaded model only when this much VRAM is free on the first GPU (else 503) |
| `before_load` | a string (run by the shell) or a list (run directly) | a command run before the model loads (to unload another server's model, say) |
| `before_load_timeout_s` | seconds; 120 | how long it may take |
| `before_load_strict` | `true` | when the command fails: always 503 (default: `min_free_vram_mib` decides; without it the model loads) |
| `before_load_ignore_failure` | `true` | when it fails: load anyway, without the VRAM check |
| `lazy_load` | `true` / `false` (default) | start the API without the model (and the image encoder); the first request loads it |
| `prevent_sleep` | `true` / `false` (default) | Windows: keep the PC awake while a request runs |
| `aliases` | a list, or a string with commas | other model names the API accepts |
| `allowed_hosts` | a host or a list; `"*"`: any; `".example.com"`: it and the names under it | the host names requests may use (also `STRATA_ALLOWED_HOSTS`) |
| `cors_origins` | an origin or a list; `"*"` allowed | web pages that may call `/v1/*` from a browser (CORS) |
| `trusted_origins` | an origin or a list, no `"*"` | web pages that count as the server's own for its settings and MCP tools |
| `mcp_servers` (or `mcpServers`) | Claude Desktop's `mcpServers` object | MCP servers whose tools a request can use when it sends `"strata_mcp": true` ([DETAILS.md](DETAILS.md#tools-from-mcp-servers)) |
| `mcp` | `timeout_s`, `start_timeout_s` (seconds), `max_result_chars`, `max_rounds` (1 or more) | limits for those tools |
| `api_monitor` | `true` / `false` (default) | the last 100 requests at `/api/requests` (kept in memory) |
| `engine_silence_s` | seconds; 300; 0: no limit | how long the engine may stay silent in a request before the server ends it (the next request starts it again) |
| `repeat_stop_tokens` | a number; 256 (on); 0: off | end a reply that repeats one token this many times in a row |
| `tool_call_recovery` | `true` / `false` (default) | accept tool calls written in other forms than the chat template's |
| `reasoning_loop_recovery` | `false` (default), `"stop"`, `"recover"` (`true` is `"recover"`) | thinking that repeats whole passages: `"stop"` ends the reply there, `"recover"` goes on from the same output with the low-effort instruction |
| `reasoning_close_retry` | `true` / `false` (default) | a reply that ends inside its thinking, with no answer and no tool call: it is continued once with the thinking closed |
| `answer_room` | `true` (default) / `false` | with `max_tokens` of 1,024 or more, thinking without a budget is closed early enough to leave room for the answer |
| `effort_position` | `"start"` (default) or `"end"` | where the effort sentence goes in the prompt (`"end"`: Strata's engine) |
| `codex_thread_titles` | `true` / `false` (default) | Codex: its thread-title requests are answered without the engine |
| `codex_compaction_cache` | `true` / `false` (default) | Codex: a compaction request is shown to the model with its conversation's tools, so its prompt starts as the cached one did |
| `env_to_user` | `true` / `false` (default) | an agent's `<env>` block (the environment it runs in) is moved from the system prompt into the first user message |
| `arch` | `qwen4exp` (default) or `qwen35moe` | the model family: the API's name without `model_name`, and the model card's sampling without `sampling` (setup writes it for Strata's engine) |
| `cuda` | `12` | the CUDA 12 engine (setup writes it) |
| `remote_expert_opt` | `false` | as setup's `--no-remote-expert-opt` (read by setup and `momoe start`) |
| `gpu_order` | `"as_given"` | keep the config's GPU order (default: an automatic layer split puts the faster card last) |
| `parallel` | 2 to 8 | Strata's engine: up to N requests decode together |
| `replicas` | a number, or `[{"gpus": [0, 1]}, {"gpus": [2, 3]}]` (each may have its own `env` and `args`) | Strata's engine: several engines, requests spread over them; `replica_cpus: false` gives them no CPU cores of their own (Linux), `replica_load_tokens` (3000) weighs a running request when choosing one |
| `slot_save_path` | a folder | Strata's engine: `POST /slots/0?action=save\|restore` (llama-server's API) to and from this folder |
| `vram_elastic` | `true` | Strata's engine: the expert cache in segments, so `POST /v1/vram` can give VRAM to other programs between requests and take it back ([DETAILS.md](DETAILS.md#sharing-the-gpu-with-other-programs-optional)) |
| `vram_segment_mib` | MiB, above 0; the engine's default 512 | Strata's engine: the segments' size |
| `split_skip_if_fits` | `true` | Strata's engine: a layer split runs on the first card alone when everything fits there |
| `expert_profile_save` | a file | Strata's engine: save what its expert cache learned |
| `expert_profile_save_every` | minutes; 10; 0: only on quitting | how often |
| `hip_ordinal` | a number | AMD, one card: the HIP device number instead of `gpu` |
| `images` | an object | image generation (Qwen-Image), [IMAGES.md](IMAGES.md); written by hand (MoMoE's setup refuses `--images`) |

## `momoe-cuda` and `momoe-metal`

The same program (`src/metal/momoe_metal.cpp`) built for CUDA or for Metal. Setup writes its arguments into the
config's `args`. The server starts it with `--serve`, picks the card(s) in the config's `"gpu"` with
`CUDA_VISIBLE_DEVICES` (so the engine's first card is device 0), adds `--layer-split` when `"gpu"` names several
cards (not with `--helper-tier`), and `--vision` for a config with pictures.

**The CPU's experts:** on `momoe-cuda`, for Nemotron (Bolt, Lightning), the Qwen3.6 architecture (Qwen3.6, Ornith,
KAT-Coder), Gemma 4 and Flint, the CPU computes the experts the GPU's pool does not hold, from RAM. Not with
`--direct-io`, `--layer-split` or `MOMOE_NO_HYBRID`; not for Flash-Next, GLM-5.3 or LFM2.5 (their missing experts are
copied to the GPU). Several settings below apply only then.

### Arguments

| Argument | Values | Engines | What |
| --- | --- | --- | --- |
| `-m` | a `.gguf` file | both | the model (its first shard) |
| `--ctx` | tokens; 32768 by default | both | the context |
| `--chunk` (or `--batch`) | tokens; 512 by default (setup writes 1024 on CUDA) | both | the prompt's chunk size. In a config write `--chunk`: the server treats `--batch` as its own (`"parallel"`) |
| `--pool-gib` | GiB | both | the expert pool's size; experts outside it come from RAM or the SSD when a token needs them. Default: `momoe-cuda` fits it to the free memory; `momoe-metal` takes every expert, made smaller if Metal's working set cannot hold it (also a `--pool-gib` pool) |
| `--slots` | a number of experts | both | the pool's size in experts instead of GiB (wins over `--pool-gib`; each group of experts gets at least a few tokens' worth) |
| `--preload` | (no value) | both | every expert of the pool loaded at the start (a pool that holds every expert does this by itself) |
| `--nocache` | (no value) | Metal | expert reads bypass macOS's file cache, as on a Mac without the RAM to cache the file |
| `--direct-io` | (no value) | CUDA | experts read from the SSD past the page cache; the CPU then computes no experts. On by itself on a unified-memory machine of 16 GiB or more (the Spark) |
| `--mtp` | drafted tokens, 1 or more (setup writes 3 or 4); none by default | both | draft with the model's MTP layer. Not for Nemotron; the file needs an MTP layer (or `--mtp-gguf`) |
| `--mtp-gguf` | a `.gguf` file | both | the MTP layer from another file (Flash-Next's GGUFs have none: setup packs one on a Mac) |
| `--draft-vocab` | a file: `data/draft_vocab_en.bin`, `data/draft_vocab_cyrillic.bin`, `data/draft_vocab_fr.bin`, or `data/draft_vocab.bin` (with Chinese, Japanese and Korean); default: every token | both | the tokens the draft head can propose |
| `--spec-min-p` | 0 to 1; 0.5 by default | both | draft only while the MTP layer gives its token at least this probability |
| `--device` | a CUDA device number; 0 by default | CUDA | the GPU, as CUDA numbers the cards it can see (the server already picks the card, so leave it out); not with `--layer-split` |
| `--layer-split` | `auto` (by each card's free VRAM), or each later card's first layer, rising: `24` or `16,32` | CUDA | layers in ranges, one per card; the CPU computes no experts. With one visible card it is ignored |
| `--helper-tier` | (no value) | CUDA | the second visible card holds a second pool under the main one's; not used when the main card holds every expert |
| `--helper-gib` | GiB; default: the helper's free VRAM less `--helper-reserve-mib` | CUDA | the helper pool's largest size |
| `--helper-reserve-mib` | MiB; 512 by default | CUDA | VRAM left free on the helper |
| `--profile` | a Strata expert profile (`data/expert-profile.bin`) | both | pin its hottest experts in the pool (`MOMOE_PIN_FRAC` of it); measured slower; not with a pool that holds every expert |
| `--nostall` | `sub` or `drop` | Metal | a token's graph runs without pauses: `sub` takes the best experts the pool holds, `drop` gives the missing ones no weight. Changes the output; not for LFM2.5 or Flint; not used when every expert is in memory ([src/metal/README.md](../src/metal/README.md#no-stall-mode---nostall-subdrop-opt-in-changes-the-output-2026-10-13)) |
| `--nostall-every` | layers, 0 or more; 0 by default | Metal | with `--nostall`: layers 0, N, 2N... keep their pause (1: the exact output) |
| `--serve` | (no value) | both | the server's protocol on stdin/stdout (what `momoe start` uses) |
| `--vision` | (no value) | both | accepted for the server; pictures always work with `--serve` |
| `--tokens`, `--tokens-file` | token ids, `1,2,3`, or a file of them | both | without `--serve`: the prompt |
| `-n` | tokens; 32 by default | both | without `--serve`: how many to write |
| `--bench-window` | (no value) | both | a test: the time of windows of 1 to 8 tokens after the `--tokens` prompt (15 runs each), then it quits |

### Environment variables: settings

Set them in the config's `env`, for example `"env": {"MOMOE_READERS": "8"}`. **On / off:** the engine checks only
whether such a variable is set: any value turns it on, `0` too. Leave it out to keep it off.

| Variable | Values | Engines | What |
| --- | --- | --- | --- |
| `MOMOE_READERS` | threads, 1 or more; 16 on CUDA, 8 on Metal, 4 on a unified-memory machine under 16 GiB | both | the threads that read experts into the pool |
| `MOMOE_DEMAND_READERS` | threads, 1 or more; none by default | both | separate threads for the reads a layer waits for, so they never queue behind the reads of predicted experts |
| `MOMOE_CPU_THREADS` | threads, 1 or more; half the hardware threads by default (at least 2) | CUDA | with the CPU's experts: the threads that compute them |
| `MOMOE_FILLERS` | threads, 0 or more; 2 by default | CUDA | with the CPU's experts: the threads that copy experts into the GPU's pool in the background (0: the readers do it) |
| `MOMOE_FILLS` | experts, 0 or more; 16 by default, no limit for Nemotron | CUDA | with the CPU's experts, through mailboxes: after each token, how many experts in the GPU's pool may be replaced by ones the CPU computed for that token. Empty slots are always filled. Each replacement is a read from RAM, which the CPU's own experts also need. 0: once full, the pool no longer changes. (Without mailboxes - `MOMOE_NO_MAILBOX`, or `--helper-tier` - there is no limit) |
| `MOMOE_ADMIT` | uses, 0 or more; 2 by default | CUDA | an expert is copied into the GPU's pool only once it was used more than this many times lately (the counts halve every 64 tokens); 0-8 measured about the same |
| `MOMOE_POLICY` | `lfu` (the default; any other value too) or `lru` | both | which expert a full pool replaces: `lfu` the least often used lately (the most recently used one is kept when two are even; measured faster), `lru` the least recently used |
| `MOMOE_PIN_FRAC` | 0 to 1; 0.6 by default | both | with `--profile`: the share of the pool pinned |
| `MOMOE_VRAM_MARGIN_MIB` | MiB; 512 by default | CUDA | VRAM the pool leaves free when the engine sizes it (no `--pool-gib` or `--slots`); with `--layer-split` on each card. Unified memory: `MOMOE_UMA_RESERVE_GIB` instead |
| `MOMOE_METAL_MARGIN_MIB` | MiB; 512 by default | Metal | the room the pool leaves in Metal's working set |
| `MOMOE_UMA_RESERVE_GIB` | GiB; 10 by default, 20% of the memory under 16 GiB | CUDA | unified memory (Spark, Jetson): memory kept free for the system |
| `MOMOE_UMA_MIN_GIB` | GiB; 24 by default, 15% of the memory under 16 GiB | CUDA | unified memory: the engine starts only with this much free besides the reserve (and `--pool-gib`), so a second engine is refused |
| `MOMOE_UMA_POOL_SHARE` | 0 to 1; 0.3 by default | CUDA | a unified-memory machine under 16 GiB (Jetson): the pool's largest share of the memory |
| `MOMOE_DIRECT` | on / off | CUDA | as `--direct-io` |
| `MOMOE_NO_DIRECT` | on / off | both | CUDA: no `--direct-io` on a unified-memory machine. Metal, every expert in memory: the router's choice is copied and looked up, as before the faster direct path |
| `MOMOE_NO_HYBRID` | on / off | CUDA | the CPU computes no experts: every expert the pool lacks is copied to the GPU |
| `MOMOE_QWEN_HYBRID` | `0` or `1` (only the first character counts; anything else: as unset) | CUDA | the CPU's experts, not Nemotron (Qwen3.6 architecture, Gemma 4, Flint): `0` copies the missing experts to the GPU instead, `1` always computes them on the CPU. Unset, the engine tries both on the first tokens and keeps the faster (and tries again when the pace changes, and every 4,096 tokens); Nemotron always computes them on the CPU |
| `MOMOE_HYB_FRAC` | 0 to 1 | CUDA | the same models: this share of the missing experts copied to the GPU (the most used first), the rest on the CPU, no trial; measured slower than `1` |
| `MOMOE_NO_MAILBOX` | on / off | CUDA | the CPU's experts through the scheduler's stops instead of mailboxes |
| `MOMOE_RMB` | on / off | CUDA | resident mailboxes (a token runs as one graph, no stops), unified memory, a pool smaller than all experts: GLM-5.3 (with `--direct-io` and `MOMOE_POOL_HOST`), LFM2.5 and Flint (with the pool in host memory) |
| `MOMOE_NO_RMB` | on / off | CUDA | resident mailboxes are on by themselves for LFM2.5 and Flint under 16 GiB without `--direct-io` (the Jetson): this turns that off, and keeps the pool in VRAM there |
| `MOMOE_POOL_HOST` | on / off | CUDA | unified memory: the pool in host memory (on by itself where resident mailboxes are, above) |
| `MOMOE_CUDA_GRAPHS` | on / off | CUDA | CUDA graphs even where the engine turns them off |
| `GGML_CUDA_DISABLE_GRAPHS` | on / off | CUDA | ggml's own switch: CUDA graphs off everywhere (the engine sets it where it turns them off) |
| `CUDA_DEVICE_MAX_CONNECTIONS` | a number; the engine sets 32 on unified memory unless it is set | CUDA | CUDA's hardware queues, so the readers' copies do not queue behind the compute |
| `MOMOE_EMBD_GPU` | on / off | CUDA | the token embeddings in VRAM (default: in RAM, as llama.cpp keeps them, when the model has its own output weights; otherwise they are in VRAM anyway) |
| `MOMOE_NO_PREDICT` | on / off | both | do not read the next layer's predicted experts ahead |
| `MOMOE_PREDICT2` | on / off | both | also predict two layers ahead (as fast, a third more read); not Nemotron or Gemma 4 |
| `MOMOE_PREDICT_K` | experts, 1 or more; all by default | both | read ahead only each token's N most likely predicted experts |
| `MOMOE_NO_PRELOAD` | on / off | both | a pool that holds every expert is not preloaded |
| `MOMOE_NO_PAUSE` | on / off | Metal | the scheduler's callback instead of pause points |
| `MOMOE_FORCE_PAUSE` | on / off | Metal | pause points even after a preload |
| `MOMOE_NO_GRAPH_REUSE` | on / off | both | build each token's graph again instead of keeping it |
| `MOMOE_NO_METAL_FIT` | on / off | Metal | do not shrink the pool to Metal's working set |
| `MOMOE_STAGE_NO_POOL` | on / off | CUDA | an expert the pool holds is read again for a prompt chunk instead of copied inside the GPU |

### Environment variables: tests and debugging

These change the output or only print; not for normal use. "On / off" as above.

| Variable | Values | Engines | What |
| --- | --- | --- | --- |
| `MOMOE_DEBUG`, `MOMOE_DEBUG_LAST` | on / off | both | tensor sums, comparable with llama.cpp's `llama-eval-callback` (`_LAST`: the last position only) |
| `MOMOE_TOP` | on / off | both | the 5 most likely tokens at each step (graphs are not kept) |
| `MOMOE_KL_DUMP`, `MOMOE_KL_REF` | a file | both | write a run's tokens and log-probabilities; feed them to another run and report KL per step |
| `MOMOE_TF_WINDOW` | tokens | both | feed the tokens of `MOMOE_TF_FILE` (a file) in windows of N and compare each window's next tokens; `MOMOE_TF_PROBS` (a file): the top 5 probabilities to it |
| `MOMOE_BENCH_ALL`, `MOMOE_BENCH_SNAP` | on / off | both | with `--bench-window`: every position's output; the state saved as for a verify window |
| `MOMOE_FORCE_CB` | on / off | both | the scheduler's callback path, no kept graphs |
| `MOMOE_NO_QSA` | on / off | both | Flash-Next's attention to every position |
| `MOMOE_FLINT_NO_RING`, `MOMOE_FLINT_NO_SCALE` | on / off | both | Flint without its sliding-window ring; without its down-projection scale |
| `MOMOE_TEST_SKIP_CPU` | on / off | CUDA | skip the CPU's experts (timing only: the output is wrong) |
| `MOMOE_CPU_PROF` | on / off | CUDA | timings of the CPU's experts |
| `MOMOE_SEGTIME` | on / off | both | GPU time between stops |
| `MOMOE_MB_DEBUG`, `MOMOE_RMB_DEBUG` | on / off | CUDA | the mailboxes' and resident mailboxes' debug lines |
| `MOMOE_METAL_FIT_DEBUG` | on / off | Metal | the Metal fit's figures |
| `MOMOE_GGML_DEBUG` | on / off | both | ggml's debug lines (one per token with CUDA graphs) |
| `MOMOE_CUDA_DUMP` | `FROM,TO` (graph call numbers) | CUDA | ggml lists the nodes of those graph calls |

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
