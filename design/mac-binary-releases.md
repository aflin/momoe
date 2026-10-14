# Ready-made Mac engine, downloaded from GitHub Releases

**Status:** Planned, for after the repository is public (2026-10-12). Nothing built yet.

## The decision

On macOS, `momoe setup` downloads a ready-made `momoe-metal` instead of compiling it. This is an exception to Momoe's
"always compile from source" (setup, 2026-10-11), which stays the rule for NVIDIA, AMD and Intel: on a Mac one binary
covers every Apple Silicon Mac, unlike CUDA, where card generations and toolkits multiply the builds. The source
build stays available (`--build`).

Why it is worth it on the Mac:

- One target: every Apple Silicon Mac (M1 on) runs the same arm64 code; no driver or toolkit matrix.
- Self-contained: the engine is one ~7 MB file (~3 MB compressed); ggml is linked in statically and it uses only
  macOS's frameworks (Metal, Foundation, Accelerate).
- No GPU compiler on the user's Mac: the Metal shaders are embedded as source (`GGML_METAL_EMBED_LIBRARY`) and macOS
  compiles them for the Mac's GPU at the first start.
- It takes the hardest part out of setup on a Mac: without it every user needs Xcode's command-line tools and CMake
  just to build the engine.

## What a distributable build needs (not done yet)

1. **Generic Apple Silicon, not the build machine's CPU.** The development build uses `GGML_NATIVE=ON`, which
   compiles for the exact CPU (the M4 here); some of those instructions an M1 lacks. Release: `GGML_NATIVE=OFF` and a
   generic arm64 target.
2. **A minimum macOS.** Set `CMAKE_OSX_DEPLOYMENT_TARGET` (macOS 14 or 15). The development build uses the macOS 27
   SDK and by default requires macOS 27. ggml checks newer GPU features at run time (residency sets, bfloat,
   tensor ops) and skips them on older systems; test that on an older macOS (the Mac Studio, if it runs one).
3. **Gatekeeper.** A file that setup's own Python code downloads is not quarantined (only browsers and Finder mark
   downloads), so an unsigned binary runs; the linker already gives arm64 binaries an ad-hoc signature. A binary
   someone downloads from a web page is blocked unless signed and notarized with an Apple developer account: not
   needed while setup does the download.
4. **Checksums.** GitHub publishes a SHA-256 digest for every release file through its API; setup checks the download
   against it before unpacking. Strata's setup had this code (`engine_digest`, `verify_engine_archive`); Momoe
   removed it with the ready-made downloads (commit 59deafa3, "setup: Momoe always compiles its engine") and it can
   come back for the Mac binary only.

## Measured: a binary built on another Mac (2026-10-13)

The momoe-metal built on en (M1 Ultra, macOS 15.6.1, SDK 15.5, minimum macOS 15.0, `-mcpu=native` for the M1) ran on
zephyr (M4, macOS 27.0) next to zephyr's own build of the same source (SDK 15.2, `-mcpu=native` for the M4).
Alternating runs with 60 s cool-downs, Ornith-1.5 IQ4_XS, 256 tokens:

| | en's binary | zephyr's own |
|---|---:|---:|
| every expert in memory | 28.2 / 28.2 tok/s | 28.2 / 28.0 tok/s |
| `--pool-gib 7 --nocache` (the rest from the SSD) | 13.0 / 13.4 tok/s | 13.5 / 13.4 tok/s |

The same 256 tokens in all 8 runs. The shaders are compiled for the local GPU at the start either way, and the CPU
code's share of the time is small, so an M1-targeted build loses nothing measurable on an M4. Not tested the other
way: zephyr's M4-native build may use instructions an M1 lacks, which is why the release builds with
`GGML_NATIVE=OFF` (point 1 above) on the oldest macOS it supports.

## Where it is downloaded from

GitHub Releases of `aflin/momoe`:

- A release (e.g. `v0.1.0-mac`) carries `momoe-metal-macos-arm64.tar.gz`, at
  `https://github.com/aflin/momoe/releases/download/<tag>/momoe-metal-macos-arm64.tar.gz`;
  `https://github.com/aflin/momoe/releases/latest/download/...` always points at the newest.
- Made with `gh release create <tag> momoe-metal-macos-arm64.tar.gz --notes "..."`, by hand from the Mac to begin
  with.
- Later, a GitHub Actions workflow on an Apple Silicon runner (`macos-14` or newer) builds and attaches it when a
  release is tagged, so releases do not depend on one Mac. Actions minutes are free for public repositories (macOS
  runners count 10x against a private repository's minutes).

## The condition: a public repository

The repository is private. Release files of a private repository need a GitHub login or token to download, so setup
on someone else's Mac cannot fetch the binary until the repository is public. Until then: build from source on the
Mac (as now), or use the owner's `gh` login for testing the download path.

## When the repository is public

1. A release build: `GGML_NATIVE=OFF`, a deployment target (14 or 15), stripped; checked on an M1-class Mac or
   an older macOS if one is at hand.
2. The first release with the binary, by hand.
3. Setup on macOS: download, check the SHA-256 against GitHub's digest, unpack into `engine/`; `--build` compiles
   instead (needs Xcode's command-line tools and CMake).
4. The Actions workflow for later releases.

Related: [mac-port.md](mac-port.md) (the Metal engine).
