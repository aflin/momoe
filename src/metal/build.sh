#!/bin/sh
# build momoe-metal with Momoe's copy of ggml (src/metal/ggml, patched by patch_ggml_pause.py)
set -e
cd "$(dirname "$0")"
export PATH="$PWD/../../.venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"   # cmake from .venv (requirements.txt)
python3 patch_ggml_pause.py ggml
B=../../build-metal/cmake
mkdir -p ../../build-metal
# MOMOE_PORTABLE=1 (the release, docker/build-macos.sh): ggml's CPU code for any Apple Silicon (the compiler's arm64
# default, the M1) instead of this Mac's, and no OpenMP (Homebrew's libomp is not on other Macs).  The oldest macOS it
# runs on is MACOSX_DEPLOYMENT_TARGET's.
PORT=""
[ -n "${MOMOE_PORTABLE:-}" ] && PORT="-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF"
cmake -S . -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo $PORT > $B.configure.log 2>&1 || { tail -20 $B.configure.log; exit 1; }
cmake --build $B -j ${JOBS:-8} > $B.build.log 2>&1 || { grep -E "error|Error" $B.build.log | head -30; exit 1; }
rm -f ../../build-metal/momoe-metal && cp $B/momoe-metal ../../build-metal/momoe-metal
