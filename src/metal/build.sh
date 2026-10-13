#!/bin/sh
# build momoe-metal with Momoe's copy of ggml (src/metal/ggml, patched by patch_ggml_pause.py)
set -e
cd "$(dirname "$0")"
export PATH="$PWD/../../.venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"   # cmake from .venv (requirements.txt)
python3 patch_ggml_pause.py ggml
B=../../build-metal/cmake
mkdir -p ../../build-metal
cmake -S . -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo > $B.configure.log 2>&1 || { tail -20 $B.configure.log; exit 1; }
cmake --build $B -j ${JOBS:-8} > $B.build.log 2>&1 || { grep -E "error|Error" $B.build.log | head -30; exit 1; }
rm -f ../../build-metal/momoe-metal && cp $B/momoe-metal ../../build-metal/momoe-metal
