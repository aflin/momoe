#!/bin/bash
# docker/build-in-oven.sh <stage> <name> -- runs INSIDE the oven (docker/build.sh starts it; the repository is /momoe).
#
#   build    setup.py --build-release in oven/<name>/momoe (the committed tree build.sh exported there): the Strata
#            engine and its image encoder (engine/), DeepSeek's (engine-dsv41/), momoe-cuda (build-cuda/) and its image
#            encoder (build-vision-momoe/), for every card in MOMOE_RELEASE_ARCHS and any AVX2 CPU; on arm64 momoe-cuda
#            and its CUDA image encoder (build-vision-momoe-cuda/) only; then RELEASE.json
#   package  that tree, without the build folders, -> dist/momoe-<version>-<name>.tar.gz (+ .sha256)
set -euo pipefail

STAGE="${1:-}"
NAME="${2:?the build name, e.g. linux-x86_64-cu13}"
WORK="/momoe/oven/$NAME"
TREE="$WORK/momoe"

source /opt/rh/gcc-toolset-13/enable
export PATH="/usr/local/cuda/bin:$PATH"
git config --global --add safe.directory '*' 2>/dev/null || true

case "$STAGE" in
  build)
    echo "==> $(gcc --version | head -1); $(nvcc --version | grep -i release); archs ${MOMOE_RELEASE_ARCHS}"
    rm -f "$TREE/RELEASE.json"
    cd "$TREE"
    STRATA_RELEASE_BUILD=1 python3 setup.py --build-release
    version="${MOMOE_VERSION:-$(cat "$WORK/describe")}"
    python3 - "$TREE/RELEASE.json" "$version" "$(cat "$WORK/commit")" "$NAME" "$MOMOE_RELEASE_ARCHS" \
        "$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')" <<'EOF'
import json, sys
path, version, commit, name, archs, cuda = sys.argv[1:]
json.dump({"version": version, "commit": commit, "build": name, "cuda": cuda, "glibc": "2.28",
           "archs": [int(x) for x in archs.split(";")]}, open(path, "w"), indent=1)
EOF
    echo "==> build OK: $TREE"
    ;;

  package)
    version=$(python3 -c "import json; print(json.load(open('$TREE/RELEASE.json'))['version'])")
    top="momoe-$version"
    out="/momoe/dist/$top-$NAME.tar.gz"
    pkg="$WORK/package"
    rm -rf "$pkg" && mkdir -p "$pkg/$top"
    # the source tree without the build folders and downloads; the engines and their BUILD.json stamps go in below
    rsync -a --exclude '/build*/' --exclude '/engine/' --exclude '/engine-dsv41/' --exclude '/third_party/llama.cpp*' \
        "$TREE/" "$pkg/$top/"
    # each engine with its BUILD.json stamp, where it was built (arm64 has momoe-cuda and the CUDA image encoder only)
    for f in engine/BUILD.json engine/momoe engine/momoe-vision engine/libstrata_kq.so \
             engine-dsv41/BUILD.json engine-dsv41/strata-dsv41 build-cuda/BUILD.json build-cuda/momoe-cuda \
             build-vision-momoe/BUILD.json build-vision-momoe/bin/momoe-vision \
             build-vision-momoe-cuda/BUILD.json build-vision-momoe-cuda/bin/momoe-vision; do
        [ -f "$TREE/$f" ] || continue
        mkdir -p "$(dirname "$pkg/$top/$f")"
        cp -p "$TREE/$f" "$pkg/$top/$f"
        # debug information out (the engines keep their symbols and the engine=X.Y.Z line setup reads)
        case "$f" in *.json) ;; *) strip --strip-debug "$pkg/$top/$f" ;; esac
    done
    # the licenses of what is compiled into the engines (MIT, all of it)
    {
        echo "# Third-party notices"
        echo
        echo "Momoe's engines in this package are compiled from Momoe's source (LICENSE, MIT) together with the code"
        echo "below, also MIT.  No NVIDIA file is in this package: the CUDA 13 runtime (cudart, cuBLAS) comes from"
        echo "your system or from NVIDIA's pip packages, under NVIDIA's license.  The C++ runtime parts linked in by"
        echo "gcc-toolset-13 are covered by the GCC Runtime Library Exception."
        for f in src/metal/ggml/LICENSE third_party/ggml/LICENSE third_party/project-maya/LICENSE \
                 third_party/libstrix/NOTICE.md; do
            [ -f "$TREE/$f" ] || continue
            echo; echo "## $f"; echo; echo '```'; cat "$TREE/$f"; echo '```'
        done
        echo; echo "## llama.cpp (ggml-org/llama.cpp, the image encoders and the tools)"; echo
        echo '```'; cat "$(ls "$TREE"/third_party/llama.cpp*/LICENSE | head -1)"; echo '```'
    } > "$pkg/$top/THIRD_PARTY_NOTICES.md"
    tar -C "$pkg" -czf "$out" "$top"
    rm -rf "$pkg"
    (cd /momoe/dist && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
    ls -l "$out"
    echo "==> package OK: $out"
    ;;

  *)
    echo "unknown stage: $STAGE  (build | package)" >&2
    exit 1
    ;;
esac
