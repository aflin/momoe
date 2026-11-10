#!/bin/sh
# build-macos.sh <stage> -- Momoe's macOS release (Apple Silicon), built on a Mac (there is no Docker oven for macOS).
#
#   build-macos.sh build     # the committed HEAD -> oven/macos<N>-arm64/momoe/: momoe-metal and its Metal image encoder
#   build-macos.sh package   # that build -> dist/momoe-<version>-macos<N>-arm64.tar.gz (+ .sha256), N the oldest macOS
#
#   Environment:
#      MACOSX_DEPLOYMENT_TARGET  the oldest macOS the engines run on (default 12.0)
#      MOMOE_VERSION             the version in the file name (default: git describe)
#      PYTHON                    a Python 3.10+ for setup.py (default: the first one found, else this checkout's own
#                                .python or .venv)
#
# Needs Xcode's command line tools (clang, the Metal framework headers), git and rsync; cmake and ninja are installed
# into oven/macos<N>-arm64/venv.  Only what is committed is built (git archive HEAD).  The engines link only macOS's own
# frameworks; their Metal shaders are compiled by macOS when they start.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "build-macos.sh runs on an Apple Silicon Mac" >&2; exit 1; }
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
NAME="macos${MACOSX_DEPLOYMENT_TARGET%%.*}-arm64"         # (CMake keeps a folder's first deployment target: one per)
WORK="$REPO/oven/$NAME"
TREE="$WORK/momoe"

find_python() {
    for c in "${PYTHON:-}" python3.14 python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3 \
             "$REPO/.python/python/bin/python3" "$REPO/.venv/bin/python"; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && \
            "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null && { echo "$c"; return; }
    done
    echo "no Python 3.10+ found: set PYTHON=/path/to/python3" >&2; exit 1
}

case "${1:-}" in
  build)
    PY=$(find_python)
    cd "$REPO"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || \
        echo "==> note: uncommitted changes are not in the release (it is built from HEAD $(git rev-parse --short HEAD))"
    # the committed tree; files whose content did not change keep their time, so only what changed is compiled again
    stage="$WORK/stage"
    rm -rf "$stage" && mkdir -p "$stage" "$TREE"
    git archive HEAD | tar -x -C "$stage"
    rsync -rlc --delete --exclude '/build*/' --exclude '/third_party/llama.cpp-vision/' \
        --exclude '/third_party/*.zip*' --exclude '/.venv/' "$stage/" "$TREE/"
    rm -rf "$stage"
    git describe --tags --always > "$WORK/describe"
    git rev-parse HEAD > "$WORK/commit"
    [ -x "$WORK/venv/bin/cmake" ] || { "$PY" -m venv "$WORK/venv" && "$WORK/venv/bin/python" -m pip install -q cmake ninja; }
    export PATH="$WORK/venv/bin:$PATH"
    echo "==> $(clang --version | head -1); macOS $(sw_vers -productVersion) building for $MACOSX_DEPLOYMENT_TARGET and newer"
    rm -f "$TREE/RELEASE.json"
    cd "$TREE"
    STRATA_RELEASE_BUILD=1 "$WORK/venv/bin/python" setup.py --build-release
    version="${MOMOE_VERSION:-$(cat "$WORK/describe")}"
    "$WORK/venv/bin/python" - "$TREE/RELEASE.json" "$version" "$(cat "$WORK/commit")" "$NAME" "$MACOSX_DEPLOYMENT_TARGET" <<'EOF'
import json, sys
path, version, commit, name, macos = sys.argv[1:]
json.dump({"version": version, "commit": commit, "build": name, "macos": macos}, open(path, "w"), indent=1)
EOF
    # every program must say the macOS it was asked for (a library built for a newer one would raise it)
    for f in build-metal/momoe-metal build-vision-momoe/bin/momoe-vision; do
        minos=$(otool -l "$f" | awk '/LC_BUILD_VERSION/{b=1} b && /minos/{print $2; exit}')
        echo "    $f: macOS $minos and newer; $(otool -L "$f" | tail -n +2 | grep -vc '/System/Library/\|/usr/lib/') libraries outside macOS"
        [ "$minos" = "${MACOSX_DEPLOYMENT_TARGET%.0}" ] || [ "$minos" = "$MACOSX_DEPLOYMENT_TARGET" ] || \
            { echo "$f is built for macOS $minos, not $MACOSX_DEPLOYMENT_TARGET" >&2; exit 1; }
    done
    echo "==> build OK: $TREE"
    ;;

  package)
    [ -f "$TREE/RELEASE.json" ] || { echo "no finished build in $TREE: run 'docker/build-macos.sh build' first" >&2; exit 1; }
    version=$(sed -n 's/.*"version": "\(.*\)".*/\1/p' "$TREE/RELEASE.json")
    top="momoe-$version"
    out="$REPO/dist/$top-$NAME.tar.gz"
    pkg="$WORK/package"
    rm -rf "$pkg" && mkdir -p "$pkg/$top" "$REPO/dist"
    rsync -a --exclude '/build*/' --exclude '/third_party/llama.cpp*' --exclude '/.venv/' "$TREE/" "$pkg/$top/"
    for f in build-metal/BUILD.json build-metal/momoe-metal build-vision-momoe/BUILD.json build-vision-momoe/bin/momoe-vision; do
        mkdir -p "$(dirname "$pkg/$top/$f")"
        cp -p "$TREE/$f" "$pkg/$top/$f"
        case "$f" in *.json) ;; *) strip -S "$pkg/$top/$f" && codesign -f -s - "$pkg/$top/$f" 2>/dev/null ;; esac
    done
    {
        echo "# Third-party notices"
        echo
        echo "Momoe's engines in this package are compiled from Momoe's source (LICENSE, MIT) together with the code"
        echo "below, also MIT.  They link only macOS's own frameworks."
        for f in src/metal/ggml/LICENSE third_party/llama.cpp-vision/LICENSE; do
            [ -f "$TREE/$f" ] || continue
            echo; echo "## $f"; echo; echo '```'; cat "$TREE/$f"; echo '```'
        done
    } > "$pkg/$top/THIRD_PARTY_NOTICES.md"
    COPYFILE_DISABLE=1 tar -C "$pkg" -czf "$out" "$top"      # (no ._ resource files in the archive)
    rm -rf "$pkg"
    (cd "$REPO/dist" && shasum -a 256 "$(basename "$out")" > "$(basename "$out").sha256")
    ls -l "$out"
    echo "==> package OK: $out"
    ;;

  ""|-h|--help)
    sed -n '2,/^set -e/{/^set -e/!p;}' "$0" | sed 's/^# \{0,1\}//' ;;
  *)
    echo "unknown stage: $1  (build | package)" >&2
    exit 1 ;;
esac
