#!/bin/sh
# build momoe-cuda (Momoe's ggml engine on an NVIDIA card, Linux: Nemotron 3.5 Bolt) into build-cuda/
#   NVCC: the CUDA compiler (default: nvcc on PATH, else /usr/local/cuda/bin/nvcc)
#   CUDA_ARCHS: the cards' architectures, e.g. "86;89" (default: native, the cards in this PC)
#   MOMOE_PORTABLE=1 (the release, docker/): ggml's CPU code for any AVX2 CPU, or on arm64 for armv8.2-a with dotprod,
#     fp16 and i8mm (Strata's STRATA_PORTABLE arm64 floor), instead of this machine's
# The first build compiles ggml's CUDA kernels: 10-20 minutes.
set -e
cd "$(dirname "$0")"
export PATH="$PWD/../../.venv/bin:$PATH"             # cmake from .venv (requirements.txt)
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
# the compiler's real path: CMake throws its whole cache away (and every object is compiled again) when the path
# changes, and /usr/local/cuda/bin/nvcc and /usr/local/cuda-13.0/bin/nvcc are the same compiler through a symlink
NVCC="$(readlink -f "$NVCC" 2>/dev/null || echo "$NVCC")"
B=../../build-cuda/cmake
PORT=""
if [ -n "${MOMOE_PORTABLE:-}" ]; then
  PORT="-DGGML_NATIVE=OFF"
  [ "$(uname -m)" = aarch64 ] && PORT="$PORT -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16+i8mm"
fi
mkdir -p ../../build-cuda
cmake -S . -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-native}" $PORT > $B.configure.log 2>&1 || { tail -20 $B.configure.log; exit 1; }
cmake --build $B -j "${JOBS:-$(nproc)}" --target momoe-cuda > $B.build.log 2>&1 || { grep -E "error|Error" $B.build.log | head -30; exit 1; }
rm -f ../../build-cuda/momoe-cuda && cp $B/momoe-cuda ../../build-cuda/momoe-cuda
