# Conversation cache: spill to a memory-mapped file when RAM is short

**Status:** Proposed (2026-10-12). Nothing here is built; numbers marked *estimate* are not measured.

## The problem

The engine holds one conversation's context (KV cache, DeltaNet state, checkpoints) on the GPU. A request that does not
start with the same tokens replaces it, and the conversation it replaced has to read its whole prompt again next time.

Claude Code with automatic permissions sends two different conversations in turn: the agent's own, and a
permission classifier with its own system prompt. Measured on Ornith-1.5-IQ4_XS, RTX 4070 Ti (12 GB), Ryzen 9 5950X,
63 GB RAM, `ornith-test.log` 2026-10-12:

| request | prompt tokens | reused | prompt time |
|---|---|---|---|
| classifier | 76,236 | 0 | 33.1 s |
| agent | 133,067 | 0 | 86.3 s |
| classifier | 76,283 | 0 | 33.0 s |
| agent | 133,208 | 0 | 86.5 s |

With `--dangerously-skip-permissions` (one conversation) each turn reused ~134,000 tokens and read 100-300 new ones in
about 1 s.

## What exists: the conversation cache in pinned RAM

The engine already has an opt-in cache that parks the inactive conversation in host RAM and copies it back on a
switch (`--conversation-cache-mib`, `--conversation-cache-slots`, `--conversation-cache-min-tokens`; Strata
docs/DETAILS.md, "Multiple conversations"). Default 0 = off.

Turned on in `ornith-test.json` (`--conversation-cache-mib 8192 --conversation-cache-slots 4
--conversation-cache-min-tokens 8192`) and measured with two different prompts sent alternately, three turns each, same
machine:

| | 37,600-token prompt | 22,100-token prompt |
|---|---|---|
| cache off | 12.0 s every turn | 6.5 s every turn |
| cache on | 12.1 s, then 0.9 s, 0.8 s | 6.8 s, then 1.2 s, 1.2 s |

- park: 106-290 ms; restore: 42-55 ms.
- snapshot size: 769,087,268 bytes for 37,644 tokens, about **20 KB per token** (KV, draft-layer KV and
  checkpoints; Ornith, `--kv int8`). Checkpoints are a fixed part, so the per-token figure is rough.
- after a restore, unchanged KV pages are kept for the next park (`reused_kv_bytes` in the log), so a park copies
  mostly what is new.

*Estimates* from that figure: a 262K conversation is ~5.3 GB; the Claude Code pair above (133K + 76K) is ~4.3 GB.
Flash-Next has a different layer layout; its cost per token is not measured.

The 8192 MiB in the test is not derived from anything: it is the size Strata's setup suggests in a tip
(`AGENT_CACHE_MIB`). The cache is a budget; the oldest parked conversation is dropped when a new one does not fit.

## The proposal

On a PC without the RAM to spare, park conversations in a memory-mapped file instead of (or after) pinned RAM.

### What it would cost

1. **A slower copy when the file is warm (in the page cache).** The GPU copies directly only from pinned memory.
   From a file mapping CUDA goes through a staging buffer, typically around half the speed. *Estimate:* a 133K
   restore (~2.7 GB) in 0.3-0.5 s instead of ~0.2 s. Against the 86 s re-read it replaces, this does not matter.
   Pinning the mapping (`cudaHostRegister`) would restore the speed but ties up the RAM again, which defeats the
   point.
2. **A disk read when the file is cold (evicted from the page cache).** *Estimate:* 2.7 GB from NVMe in 0.5-1 s.
   Still far below a re-read.
3. **Disk writes.** Every park writes the snapshot. With automatic permissions Claude Code switches every turn, so
   this could be several GB per turn and hundreds of GB on a long day of work: SSD wear. Keeping unchanged pages
   between parks (as the RAM cache already does) reduces it but does not remove it.
4. **Competition with the experts.** On a RAM-limited PC the model's experts are memory-mapped too (the file tier,
   `--mmap-experts`). Snapshot pages and expert pages share the same page cache: keeping a parked conversation can
   evict experts and slow every token. This is likely the largest cost, and the one to measure first.

Costs 3 and 4 are why the file should be the overflow, not the default.

### Design

1. **RAM first, sized from the model.** The default budget comes from the model's measured bytes per token, the
   context, and the number of conversations to keep (two covers an agent plus its classifier), capped by the RAM free
   after the model. Not a fixed 8 GB.
2. **The file only for what does not fit.** When a park would evict a conversation from RAM, write that one to the
   file instead of dropping it. The engine already saves conversations to disk (`POST /slots/0?action=save`,
   `--session-min-free-mib`), so the snapshot format and the checks exist.
3. **Keep the long one in RAM.** With `--conversation-cache-min-tokens`, short side requests do not take a slot; the
   same rule should decide what goes to the file: the conversation least likely to come back next.
4. **Limit the writes.** Write only pages that changed since the last write of that conversation, and stop spilling
   (fall back to re-reading) past a daily write budget.

### What to measure before building it

- Restore time at 133K and 262K from pinned RAM, from a warm file and from a cold file, on the 4070 Ti.
- Decode speed with a large parked file beside `--mmap-experts` on a PC with little free RAM (or a cgroup memory
  limit on this one), against the same run with no parking: does the page cache drop experts?
- Bytes written per turn in a real Claude Code session with automatic permissions.
- The bytes per token of Flash-Next's snapshot.

## Alternative not chosen: several conversations resident in VRAM

llama.cpp's server keeps several slots in one shared KV buffer on the GPU (`-np`; one `-c` sized cache that the slots
share as needed). The engine's `--slots N` is different: each slot gets its own full cache, for decoding several
requests at once.

A shared VRAM pool would avoid the copies, but on a 12 GB card the VRAM it uses comes out of the expert cache, whose
hit rate sets the decode speed on every token. *Estimate:* the 76K classifier alone is ~1.5 GB of the expert cache. RAM
parking costs about a quarter of a second, and only on a switch. Worth revisiting on 24 GB and larger cards.
