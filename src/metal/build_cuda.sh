#!/bin/sh
# build momoe-cuda (Momoe's ggml engine on an NVIDIA card, Linux: Nemotron 3.5 Bolt) into build-cuda/
#   NVCC: the CUDA compiler (default: nvcc on PATH, else /usr/local/cuda/bin/nvcc)
#   CUDA_ARCHS: the cards' architectures, e.g. "86;89" (default: native, the cards in this PC)
# The first build compiles ggml's CUDA kernels: 10-20 minutes.
set -e
cd "$(dirname "$0")"
export PATH="$PWD/../../.venv/bin:$PATH"             # cmake from .venv (requirements.txt)
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
B=../../build-cuda/cmake
mkdir -p ../../build-cuda
cmake -S . -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-native}" > $B.configure.log 2>&1 || { tail -20 $B.configure.log; exit 1; }
cmake --build $B -j "${JOBS:-$(nproc)}" --target momoe-cuda > $B.build.log 2>&1 || { grep -E "error|Error" $B.build.log | head -30; exit 1; }
rm -f ../../build-cuda/momoe-cuda && cp $B/momoe-cuda ../../build-cuda/momoe-cuda
