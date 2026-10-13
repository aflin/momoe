# Momoe design notes

Design choices made for Momoe, proposed or implemented, one file each. `docs/` is Strata's and changes with every merge
from upstream; these are Momoe's own and stay out of it.

Each note starts with its status:

- **Proposed**: written up, not built.
- **Accepted**: decided, being built.
- **Implemented**: in the code; the note names the commit.
- **Rejected**: decided against; the note says why, so it is not proposed again.

Keep the style of the docs: plain words, measured numbers with what they were measured on, and estimates marked as
estimates.

| Note | Status | Summary |
|---|---|---|
| [conversation-cache-file-spill.md](conversation-cache-file-spill.md) | Proposed | When RAM is short, park conversations in a memory-mapped file instead of pinned RAM |
| [second-gpu-layer-split.md](second-gpu-layer-split.md) | Rejected (this PC) | An RTX 3060 beside the 4070 Ti for Flash-Next: more cache hits, no faster writing |
| [mac-port.md](mac-port.md) | In development | Momoe on Apple Silicon: a Metal engine reading experts from the SSD into a pool |
| [mac-binary-releases.md](mac-binary-releases.md) | Planned (after the repo is public) | The Mac engine as a ready-made download from GitHub Releases |
