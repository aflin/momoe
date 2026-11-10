# Momoe's Linux release oven

Builds Momoe's Linux release, a tar.gz with every engine already compiled, inside a
[manylinux_2_28](https://github.com/pypa/manylinux) container with the CUDA 13.0 toolkit. A release
built this way runs on glibc 2.28 and newer: Ubuntu 20.04+, Debian 11+, RHEL / Alma / Rocky 8+, Fedora
and Arch. CUDA 13 needs glibc 2.28, so there is no older floor. The layout follows rampart-langtools'
`docker/` oven. Momoe needs no CPU BLAS, so the image has no OpenBLAS layer.

```
docker/build.sh build      # the committed HEAD -> oven/linux-<arch>-cu13/momoe/, every engine compiled there
docker/build.sh package    # -> dist/momoe-<version>-linux-<arch>-cu13.tar.gz and its .sha256
docker/build.sh shell      # a shell in the oven
docker/build.sh save-image # the oven image -> oven/momoe-oven-cu13-<arch>.image.tar.gz
```

`--rebuild-image <stage>` builds the oven image afresh first (after a Dockerfile edit). `JOBS` caps
momoe-cuda's parallel compiles. `OVEN_NETWORK=host` runs the image build and the oven on the host's
network, for a machine whose containers cannot resolve names on docker's default bridge (mo).

The macOS release has its own script, `docker/build-macos.sh` (no Docker): see [macOS](#macos).

## What is in the tar.gz

The source tree at HEAD (only what is committed), with:

| Folder | Engine | Built for |
|---|---|---|
| `engine/` | `momoe` (Strata's engine), `libstrata_kq.so`, `momoe-vision` (its GPU image encoder) | the families on upstream's own path (Flash-Next, Qwen3.6, Ornith, ...) |
| `engine-dsv41/` | `strata-dsv41` | DeepSeek-V4.1-Flash |
| `build-cuda/` | `momoe-cuda` | the ggml families (GLM, MiniMax, MiMo, Ling, Gemma, ...) |
| `build-vision-momoe/bin/` | `momoe-vision` (CPU image encoder) | pictures on the ggml families |
| `build-vision-momoe-cuda/bin/` | `momoe-vision` (CUDA image encoder, aarch64 only) | pictures where the GPU shares the memory |

On aarch64 every family runs on momoe-cuda (on the DGX Spark the GPU shares the memory), so that
tar.gz holds `build-cuda/` and the CUDA image encoder only.

Each folder's `BUILD.json` is the stamp setup.py writes when it compiles an engine itself, marked
`"release": true`. setup.py finds the engines built for this source and does not compile them, so a PC
needs no compiler and no CUDA toolkit. It compiles anyway, as before, in these cases:

- a card outside the release's architectures;
- a CPU without AVX2;
- a `git pull` or edit that changes the engines' source.

`RELEASE.json` names the version, commit, CUDA version and architectures; `THIRD_PARTY_NOTICES.md` holds
the licenses of the code compiled into the engines (all MIT).

## Targets

| Host | Default `MOMOE_RELEASE_ARCHS` | Cards |
|---|---|---|
| x86_64 | `75;80;86;89;90;100;120` | RTX 20 to RTX 50, A100, H100, B200 |
| aarch64 | `90;100;110;120;121` | GH200, GB200, Thor, RTX PRO, DGX Spark (GB10) |

The CPU code is portable:

- x86_64: any AVX2 CPU (`STRATA_PORTABLE`, and `GGML_NATIVE=OFF` for momoe-cuda);
- aarch64: armv8.2-a with dotprod, fp16 and i8mm.

The Jetson (Orin, sm_87) is not a target: it compiles from source.

## The CUDA runtime

The tar.gz holds no NVIDIA file. On the user's PC, setup.py (`release_cuda_dirs`) looks for the CUDA 13
runtime the engines load (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`), in this order:

1. this PC's own: on the loader's path (`ldconfig -p`) or in a CUDA 13 toolkit;
2. NVIDIA's pip packages (`nvidia-cublas`, `nvidia-cuda-runtime`), installed into `.venv`;
3. otherwise setup stops and says what to install.

The NVIDIA driver must be 580 or newer (CUDA 13). `libcuda.so.1` always comes from the driver.

## Mounts and the build tree

The repository is mounted at `/momoe` and the oven runs as your user. `build` exports HEAD to
`oven/linux-<arch>-cu13/momoe/` with rsync, which leaves files whose content did not change as they
were, so a rebuild compiles only what changed. The build folders and the llama.cpp downloads stay
between builds. `oven/` and `dist/` are ignored by git.

The first image build downloads the CUDA toolkit: the image is about 10 GB. The first engine build
compiles every architecture: about an hour or more on a 16-core machine.

## macOS

```
docker/build-macos.sh build     # on an Apple Silicon Mac: the committed HEAD -> oven/macos12-arm64/momoe/
docker/build-macos.sh package   # -> dist/momoe-<version>-macos12-arm64.tar.gz and its .sha256
```

It builds momoe-metal and its Metal image encoder through `setup.py --build-release`, for
`MACOSX_DEPLOYMENT_TARGET` (default 12.0, checked in each program afterwards). The CPU code is for any
Apple Silicon (the M1 baseline) with no OpenMP. The programs link only macOS's own frameworks, and
macOS compiles their Metal shaders when they start. Each deployment target has its own work folder
and file name (`macos12-arm64`), because CMake keeps a folder's first target. It needs Xcode's command
line tools, git, rsync and a Python 3.10+; cmake and ninja go into the work folder's venv.

macOS 12 is the floor. Built for 11.0, the image encoder's Metal code (llama.cpp's ggml-metal)
calls `waitUntilSignaledValue:timeoutMS:`, which is macOS 12+. momoe-metal itself has nothing newer
than macOS 11. The encoder's ggml BLAS backend (Accelerate's new `cblas_sgemm`, macOS 13.3+) is
left out of the release; the encoder runs on Metal.

A browser's download marks every file as downloaded, and macOS would stop the engines, which are not
signed with a developer ID. `setup.sh` clears the mark in a release folder (one with `RELEASE.json`).
Signing and notarizing with an Apple Developer ID would make that step unnecessary.

Tested (2026-11-09) by unpacking the macos12 tar.gz into an empty folder with its own HOME and running
setup with `--gguf-dir`. Nothing was compiled, and Ornith-1.5 IQ4_XS answered:

- en (M1 Ultra, macOS 15.6.1): 66.8 tok/s; the Metal image encoder read a test picture correctly;
- zephyr (M4, macOS 27): a 6 GiB pool with the rest of the experts streamed from the SSD, 13.4 tok/s.

No Mac with macOS 12 to 14 was available to test.
