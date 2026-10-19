# CUDA1–3 expert experiment

CUDA0 keeps the dense layers, KV/state, MTP and its existing expert cache. Up to
three independent expert caches fill CUDA1, CUDA2 and CUDA3 in that order. Each
expert has one owner: CUDA0, one secondary GPU, or the CPU pool. Each secondary
GPU computes its rows while the CPU pool works; the results return through
private pinned host buffers before CUDA0 continues the layer. The GPU launches
can overlap, but each layer waits for all three.

The shipped profile ranks 8,000 `(layer, expert)` pairs. CUDA0 takes its usual
`--expert-cache auto` allocation. With only CUDA1, it takes the first remaining
pairs. When CUDA2 or CUDA3 is enabled, the remaining profiled pairs are assigned
round robin to the secondary GPUs. The ranked list is then extended with all
other pairs in expert-then-layer order and distributed in the same way. This
fills the additional VRAM and lets all secondary GPUs see frequent experts, but
unranked experts may hardly ever be routed. The startup
log reports slots and GiB per card; each request reports the number of expert
entries actually computed on each secondary GPU. Remote outputs are packed
before returning over PCIe/USB4: the request log compares MiB transferred
with the previous full-row transfer. The input and routing metadata use pinned
host staging. This reduces transfer traffic and launch overhead, but does not
release the expert arena in system RAM or significantly change VRAM use.

The optional `--gpu-placement layer` gives each layer to exactly one secondary
GPU (layer number modulo the number of enabled cards). It fills that card with
the most frequently routed experts of its layers first, then the unranked ones.
This can reduce USB4 device switches and synchronization when several secondary
cards serve the same layer in the default `stripe` mode. All three cards still
keep their requested expert slots; the primary CUDA0 cache and CPU fallback
continue to work. The log counts active layer launches on each card, so the
tradeoff can be measured alongside tokens per second. This placement is
experimental and may be slower if a given layer needs more than one card's
compute capacity.

The experiment requires the corresponding visible CUDA devices and a CUDA
enabled build. Each card must keep at least 512 MiB free. Tiers must be enabled
in order. Under WDDM (Windows, WSL2), CUDA registration of the host expert arena is capped at
8 GiB to leave room for the contexts and MTP on CUDA0 (`STRATA_ARENA_PIN_GIB` overrides it; on Linux the whole
arena is registered). The rest remains
available to the CPU pool; the PCIe expert path is available only for the
registered layers. Without secondary GPUs, the original uncapped registration
behavior applies.

On an existing Linux installation, apply the patch to its source tree, then run:

```sh
./setup.sh --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

On Windows, from PowerShell in the installation folder:

```powershell
.\START-HERE.bat --setup --gpu1-experts 5000 --gpu2-experts 5000 --gpu3-experts 5000
```

To test the layer placement instead, add `--gpu-placement layer` to the setup
command. To go back, run the same command with `--gpu-placement stripe`. After
the new engine is built, the mode can also be changed without a rebuild: edit
`--expert-cache-remote-placement` in `strata-iq3_xxs.json` from `layer` to
`stripe` or vice versa and restart the server. If the argument is absent, the
default is `stripe`.

Choose the same model, size, context and image settings as your current
installation when prompted. Setup recompiles the engine, updates its server
configuration, and starts it. Later starts keep those settings. To tune the
slots without recompiling, edit the values after `--expert-cache-device1`,
`--expert-cache-device2` and `--expert-cache-device3` in the stored
`strata-iq3_xxs.json`, then restart with `.\run-iq3_xxs.bat`.

Compare identical requests at one, two and four GPUs, preferably with several
repeats. VRAM use alone does not show useful offload: compare the per-request
CUDA1–3 counts and tokens per second. More GPU contexts and synchronization
may lower the speed. Prompt prefill still uses CUDA0; secondary GPUs serve
decode, including MTP verification. No peer-to-peer access is required.

Longer contexts reserve more KV/state memory on CUDA0, which reduces its
automatic expert cache. On a 64 GB PC `setup.py` limits IQ3_XXS to 128K even
if 262K was selected; the secondary caches still require the host expert arena.
To try 262K on 64 GB, choose Q2_0 or IQ2_XS instead and check that CUDA0 still
has enough free VRAM after loading. Benchmark long prompts separately from
short decode requests.

The existing warning about the CUDA0 expert-cache GPU hit path still applies:
its outputs diverge from cache-off runs. Treat performance as experimental
until the generated tokens have been validated.

## Optional helper decode optimization

`--remote-expert-opt` (`--serve` only) optimizes the CUDA1-3 helper caches
above. The engine's default is off; since 0.1.39b setup adds it to a config on
two or more GPUs (`--gpus`, or "use both" at start). It acts only when a helper
cache is configured; a layer split runs exactly as before. To leave it out:
setup's `--no-remote-expert-opt`, or `"remote_expert_opt": false` in the
model's `strata-*.json` (kept when setup runs again). Measured by the PR's
author: dual RTX 4090 +63% mixed / +132% code decode over the plain helper
path; RTX 5090 + 4090 +28% / +63%. The primary cache avoids admitting experts already held
by a helper, and helpers replace cold experts with frequently routed CPU
misses using their existing same-layer slots. Each helper reduces its expert
outputs to a weighted partial sum on its GPU before returning one vector per
token. Tokens with no CPU expert work skip CPU activation quantization.

For an existing server configuration with CUDA0 as the primary and CUDA1 as a
helper, use these engine arguments alongside the model and profile arguments:

```text
--expert-cache auto --expert-cache-device1 auto --remote-expert-opt
```

`--expert-cache-device1`, `--expert-cache-device2` and
`--expert-cache-device3` now also accept `auto`: fill each helper from its
assigned ranking using actual aligned expert bytes and free VRAM, retaining
the existing 512 MiB allowance. Explicit numeric budgets still work. This is
startup capacity sizing, not throughput balancing; with several helpers a
card's truncated candidate tail is not redistributed to another card.

The optimization uses the existing host scheduling and pinned-host transport,
not P2P or tensor parallelism. Each layer still waits for its participating
helpers. It changes floating-point summation order, so enabled output is not
claimed to be bitwise identical. Only two-card CUDA operation has been measured
by its author; a community machine measured two cards on HIP (2x RX 6900 XT,
PCIe 4.0 x8 each, IQ3_S): there the expert plan's PCIe share had been taking
experts the helper already held, which made the primary card wait 43 ms per
verify window instead of 19 (decode 40 instead of 66 tok/s). #854 keeps a
helper's experts out of that share; before it, `--pcie-frac 0` avoided the
cost ([bench/results/2026-10-04-rdna2-helper-pcie-share](../bench/results/2026-10-04-rdna2-helper-pcie-share/README.md)).
Three/four cards have not been validated. Without the switch, the
existing decode path remains in use. This does not optimize the separate
`--peer-device` path below or change its existing incompatibility with helper
caches.

## Peer tier (`--peer-device`)

`--peer-device N` puts a second adaptive expert cache on CUDA device N. It
takes the ranked pairs CUDA0's cache does not hold, as many as fit. The peer
computes the rows of its own experts, for decode windows and for prompt
chunks; the activations and the results cross NVLink or another P2P path. The
tier adapts while the server runs, like the primary cache. It is an
alternative to the CUDA1-3 caches above, not a third tier beside them.

- `--peer-device N` (default off): enable the tier on CUDA device N (N >= 1).
- `--peer-reserve-mib M` (default 600): leave M MiB free on the peer card; the
  cache takes what remains. The prompt-path buffers need this headroom.
  Keep a few hundred MiB free on both cards beyond this: the prompt path loads its kernels and cuBLAS on first use
  (a start with 5 MiB left on the peer and 485 MiB on the primary stopped at the first prompt with "cuBLAS status 14").
- Whether the two cards can really exchange data is tested once at start (#2666): two RTX A4000 behind a host
  bridge reported peer access, enabled it, and `cudaMemcpyPeerAsync` then copied nothing, so the peer computed on
  zeros and the first prompt came back NaN. A test copy each way now decides; if it arrives wrong the log says so
  and the prompt path uses the host route (the one without P2P: activations and rows through mapped host memory).
  `STRATA_PEER_NO_P2P=1` asks for that route by hand.
- `--peer-slots N` (default 0): cap the tier at N experts; 0 = as many as fit.
- `--peer-adapt-swaps N` (default -1): swaps per adaptive round on the peer;
  -1 uses the primary's `--adapt-swaps`.
- `--peer-prefill-rows N` (default -1): the share of each prompt chunk's rows
  the peer computes; -1 is half of chunk x top-k, 0 keeps prompt rows on the
  primary.

Through the server, list both cards in the config's `"gpu"` (e.g. `[0, 1]`, numbered as nvidia-smi numbers them)
and add `--peer-device 1` to its `"args"`; the number is the card's position in that list (with `"gpu": [2, 0]`,
`--peer-device 1` is nvidia-smi's card 0). Several GPUs in `"gpu"` are otherwise a layer split: the server adds
`--layer-split` only when `--peer-device` is not among the args.

`--peer-device` requires `--expert-profile` and an enabled expert cache, and
the device must be visible; it refuses otherwise. It also refuses
`--layer-split` (a different second-GPU mode: use one or the other) and
`--expert-cache-device1..3` (the peer tier already caches experts on that card):

    strata generate: --peer-device cannot be combined with --layer-split (use one or the other)
    strata generate: --peer-device cannot be combined with --expert-cache-device1..3 (the peer tier already caches experts there)

Without `--peer-device` the binary is unchanged; its output is byte-identical
to the release. With `--peer-device` and the same expert set split across the
two cards, the generated tokens are byte-identical to the single-GPU run under
the exactness gate.

With a peer the prompt path keeps the MMQ path; the fused int8 prompt path is
not yet combined with the peer's rows. Mapped host buffers gain
`cudaHostAllocPortable` only with a peer, since only then does a second
context write them. The tier size is manual for now (`--peer-reserve-mib`,
`--peer-slots`); automatic sizing on small cards wants the buffer lending of
#216 and is a follow-up.

## The resident RAM mode with the helper caches

`--resident-experts` (the low-RAM mode) works with the helper caches: the
experts a helper holds are left out of the RAM copy as a split stage's are, and
the helpers' refills read through the source - the RAM copy when it holds the
bytes, the files otherwise. Measured here on an RX 6800 (gfx1030, 16 GB) + RX
6600 (gfx1032, 8 GB), Windows 11, an i5-11400F and 32 GB of RAM (the 6800 alone:
~37.5 decode tok/s, the same worker configuration): the 6600's 3.4-3.8k-expert
cache lifted decode to ~40.6 / 44.2 tok/s on 8K-token requests at 3800 slots,
with the CPU pool's share of the experts down from ~2.9 to ~0.4 entries per
layer-window. The helper's rows cost about 0.3-0.6 ms per layer on the pool's
critical path - usually less than the CPU rows and page faults they replace,
which is why the 6600 (about 2-3x the 6800's cost per row) still pays off as a
cache tier.

**Size a helper against the card's *real* free VRAM.** `--expert-cache-device1
auto` reads `cudaMemGetInfo`, which on Windows reports the *process's* WDDM
budget: it cannot see the desktop, the vision encoder (a separate process), a
streaming host or other apps on that card.  A helper auto-sized on a card that
also drives the display or the encoder over-commits, WDDM backs the excess with
system RAM ("shared GPU memory"), and that memory leaves the machine's RAM - on
a 32 GB PC it pushed the resident complement into the pagefile and made the
helper read its own cache over PCIe.  An explicit count
(`--expert-cache-device1 3800` here, of ~4100 that fit an empty card) kept the
cache in real VRAM; freeing another app's sticky allocation on the same card
was worth about the same again.  Prefer explicit counts on a card that shares
its VRAM with anything else.

## The helper tier (`--helper-tier`)

For two different NVIDIA cards, where the slower one would hold only a few layers of a split. The main card runs the
model as it would alone; the other card is the **second tier of experts under the main card's cache**, and the RAM
copy holds the rest. Each expert is in exactly one place:

1. the main card's cache (computed there);
2. the helper card's cache (computed there; the rows come back through pinned RAM, no P2P needed);
3. the RAM copy (`--resident-experts`: computed by the CPU, or read over PCIe);
4. the model file, only before it is loaded.

When the main card's cache follows the conversation it promotes an expert from the helper, and the expert it evicts
goes down to the helper's slot; the helper's own evictions go down to RAM, into the place the expert it takes in
leaves. The experts the helper holds need no RAM, and when a prompt is read the helper computes their rows itself.

**One switch.** `--helper-tier` in the config's `"args"`, with both cards in `"gpu"`, the main one first:

```json
"gpu": [1, 0],
"args": [..., "--helper-tier"]
```

It turns on `--expert-cache-device1 auto`, `--remote-expert-opt`, the resident RAM mode (unless the config chose
`--mmap-experts` or `--resident-budget-gib`), and the settings below. With `--helper-tier` the server adds no
`--layer-split` and keeps the cards in the config's order. Setup offers it when two different cards can share the
model (`--helper-tier` at setup: the two best, or the ones `--gpus` names).

**The limits on the helper card:**
- `--helper-gib G`: at most G GiB of its VRAM for experts (default: what fits).
- `--helper-reserve-mib N`: what its cache leaves free. Default 1600 with `--helper-tier`, for the prompt buffers
  the helper needs to compute prompt rows, and 512 without it.
- `--expert-cache-device1 N`: a fixed number of experts instead.

**Each setting on its own.** `--helper-tier` sets these only where the environment does not, so any of them can be
changed in the config's `"env"` (`"env": {"STRATA_HELPER_SWAPS": "8"}`), or `0` to turn one off:

| variable | `--helper-tier` | what it does |
|---|---|---|
| `STRATA_HELPER_EXCHANGE` | 1 | the helper's evictions take their replacements' places in RAM (without it they are left only in the model file) |
| `STRATA_HELPER_PROMOTE` | 1 | the main cache promotes from the helper and evicts into it (without it, the helper's experts can never move up) |
| `STRATA_HELPER_PROMOTE_MAX` | 32 (unset) | at most this many promotions and rotations a round |
| `STRATA_HELPER_ASYNC` | 1 | the helper's own swaps are queued on its stream, not waited for |
| `STRATA_HELPER_SWAPS` | 16 | at most this many of the helper's own swaps a round (the default, `--adapt-swaps` 96, held up its rows) |
| `STRATA_HELPER_LEND` | 1 | the prompt path's borrowed slots keep their experts in RAM (~3.4 GiB on a 12 GB main card) |
| `STRATA_PREFILL_HELPER_SOURCE` | 1 | the prompt path copies an expert the helper holds from the helper, not from the file |
| `STRATA_PREFILL_HELPER_ROWS` | 1 | the helper computes the prompt rows of the experts it holds |
| `STRATA_READ_CAUSES` | (unset) | a diagnostic: each request's file reads, by what they were for |

Without `--helper-tier` and these variables nothing changes.

**Measured** on an RTX 4070 Ti (main) + RTX 3060 12 GB (helper, also drives the desktop), Ryzen 9 5950X, 63 GB of
RAM, Linux, Qwen3.8-Flash-Next IQ3_XXS, 64K context, int8 KV, `--mtp`; `tools/strata_bench.py`, 3 runs per test,
greedy, 2026-10-18:

| | 4070 Ti alone (two runs) | `--helper-tier` |
|---|---:|---:|
| experts on the GPUs | 2,910 | 2,825 + 5,492 on the 3060 (8.83 GiB) |
| experts in RAM | ~41 GiB (all of them) | 29.9 GiB |
| the PC's available memory | 2.1 GB | 11.8 GB |
| short chat, decode tok/s | 58.2 / 55.9 | 59.7 |
| agent session, decode tok/s | 63.7 / 63.8 | 68.3 |
| 4K-token prompt, tok/s | 1,178 | 1,478 |
| 16K-token prompt, tok/s | 2,130 | 1,813 |
| agent turn, first token | 3.37 s | 2.51 s |

The same configuration moved 64.5-71.1 tok/s between runs in the agent session, so decode is about the same as the
main card alone; prompts up to ~4K tokens are faster and 16K-token prompts slower. Draft acceptance was 61-63% in
every helper-tier run (60-61% alone). The output is not bit-identical to the main card alone (the experts are
computed on other devices). Not measured: other cards or models, two equal cards (a layer split should suit them
better: each card runs its own layers, with no wait for the other card inside a layer), Windows.

What the separate settings did, in the same benchmark (decode short chat / agent, prompt 4K / 16K):

| step | decode | prompt | file reads |
|---|---|---|---|
| helper cache + `--resident-experts`, nothing else | 42.6 / 69.8 | 930 / 1,413 | 482 GB |
| + exchange, promotion, async, 16 swaps a round | 61.5 / 71.1 | 854 / 1,332 | ~386 GB (327 GB of them prompts) |
| + the helper as the prompt's source, the lend region in RAM | 61.3 / 64.5 | 1,025 / 1,559 | 66 GB |
| + the helper computes its prompt rows (`--helper-tier`) | 59.7 / 68.3 | 1,478 / 1,813 | 65 GB |

`--peer-device` was tried on the same pair and kept the 3060's experts in RAM too (38.7 GiB), and its prompt buffers
did not fit beside its cache.

## Choosing the primary device on Windows (`STRATA_PRIMARY_DEVICE`)

The engine runs the model on the device the runtime numbers 0.  Windows' HIP
runtime enumerates the cards in a fixed order that `HIP_VISIBLE_DEVICES` only
filters - it cannot reorder - so on a PC whose faster card comes second (an RX
6800 after an RX 6600, say) a run with both visible could only put the model on
the slow one.  `STRATA_PRIMARY_DEVICE=N` picks another visible ordinal as the
primary; unset or 0 is upstream's behavior.

- the helper caches (`--expert-cache-device1..3`) take the visible devices
  other than the primary, in order;
- the peer tier's cross-device checks use the same ordinal;
- a layer split is refused with this set (its later stages must be > 0, and
  stage 0 is the primary);
- NVIDIA cards do not need it: `CUDA_VISIBLE_DEVICES` with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` already reorders.

Measured on RX 6800 + RX 6600, Windows 11; the report accompanies this change.
