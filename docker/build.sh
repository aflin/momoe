#!/bin/sh
# build.sh <stage> -- Momoe's Linux release, built in a manylinux_2_28 + CUDA 13 "oven" (docker/Dockerfile).
#
#   build.sh build        # the committed HEAD -> oven/linux-<arch>-cu13/momoe/, every engine compiled there
#   build.sh package      # that build -> dist/momoe-<version>-linux-<arch>-cu13.tar.gz
#   build.sh shell        # an interactive shell in the oven
#   build.sh save-image   # the oven image -> oven/momoe-oven-cu13-<arch>.image.tar.gz
#
#   Flags:
#      --rebuild-image <stage>   # build the oven image afresh first (after a Dockerfile edit)
#
#   Environment:
#      MOMOE_RELEASE_ARCHS   the CUDA architectures (default: x86_64 75;80;86;89;90;100;120 = RTX 20 to RTX 50 and
#                            the A100/H100/B200; aarch64 90;100;110;120;121 = GH200, GB200, Thor, RTX PRO, DGX Spark)
#      MOMOE_VERSION         the version in the file name (default: git describe)
#      JOBS                  momoe-cuda's parallel compiles (default: all cores)
#      OVEN_NETWORK          docker's network for the image build and the oven, e.g. host where containers on the
#                            default bridge cannot resolve names (mo)
#
# Only what is committed is built (git archive HEAD); a dirty tree is named, not included.  The Jetson (Orin, sm_87)
# is not a release target: it compiles from source.  The tar.gz holds no NVIDIA file: setup.py finds the CUDA 13
# runtime on the PC, else installs NVIDIA's pip packages, else says what to install.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  CUDA_ARCHDIR=x86_64; ARCHS_DEFAULT="75;80;86;89;90;100;120" ;;
    aarch64) CUDA_ARCHDIR=sbsa;   ARCHS_DEFAULT="90;100;110;120;121" ;;
    *)       echo "no release for $ARCH" >&2; exit 1 ;;
esac
IMG="momoe-oven-cu13-$ARCH"
NAME="linux-$ARCH-cu13"
WORK="$REPO/oven/$NAME"

build_image() {
    echo "==> building the oven image '$IMG' (manylinux_2_28 + CUDA 13.0, $ARCH)..."
    docker build ${OVEN_NETWORK:+--network "$OVEN_NETWORK"} --build-arg ARCH="$ARCH" --build-arg CUDA_ARCHDIR="$CUDA_ARCHDIR" -t "$IMG" "$HERE"
}
ensure_image() {
    docker image inspect "$IMG" >/dev/null 2>&1 && { echo "==> using the oven image '$IMG'"; return; }
    tar="$REPO/oven/$IMG.image.tar.gz"
    [ -f "$tar" ] && { echo "==> restoring '$IMG' from $tar"; docker load -i "$tar" && return; }
    build_image
}

# the committed tree into $WORK/momoe; files whose content did not change keep their time, so only what changed is
# compiled again (the build folders, engines and llama.cpp downloads stay)
export_source() {
    cd "$REPO"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || \
        echo "==> note: uncommitted changes are not in the release (it is built from HEAD $(git rev-parse --short HEAD))"
    stage="$WORK/stage"
    rm -rf "$stage" && mkdir -p "$stage" "$WORK/momoe"
    git archive HEAD | tar -x -C "$stage"
    rsync -rlc --delete \
        --exclude '/build*/' --exclude '/engine/' --exclude '/engine-dsv41/' \
        --exclude '/third_party/llama.cpp/' --exclude '/third_party/llama.cpp-vision/' --exclude '/third_party/*.zip*' \
        "$stage/" "$WORK/momoe/"
    rm -rf "$stage"
    git describe --tags --always > "$WORK/describe"
    git rev-parse HEAD > "$WORK/commit"
}

oven_run() {  # $@: the command in the oven
    docker run --rm $DOCKER_TTY ${OVEN_NETWORK:+--network "$OVEN_NETWORK"} \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e MOMOE_RELEASE_ARCHS="${MOMOE_RELEASE_ARCHS:-$ARCHS_DEFAULT}" \
        -e MOMOE_VERSION="${MOMOE_VERSION:-}" -e MOMOE_ARCH="$ARCH" -e JOBS="${JOBS:-}" \
        -v "$REPO:/momoe" -w /momoe \
        "$IMG" "$@"
}

if [ "${1:-}" = "--rebuild-image" ]; then
    shift
    build_image
fi

case "${1:-}" in
    build)
        ensure_image
        export_source
        oven_run /momoe/docker/build-in-oven.sh build "$NAME" ;;
    package)
        ensure_image
        [ -f "$WORK/momoe/RELEASE.json" ] || { echo "no finished build in $WORK: run 'docker/build.sh build' first" >&2; exit 1; }
        mkdir -p "$REPO/dist"
        oven_run /momoe/docker/build-in-oven.sh package "$NAME" ;;
    save-image)
        docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image '$IMG' is not built yet" >&2; exit 1; }
        mkdir -p "$REPO/oven"
        echo "==> saving $IMG to oven/$IMG.image.tar.gz"
        docker save "$IMG" | gzip > "$REPO/oven/$IMG.image.tar.gz" ;;
    shell)
        ensure_image
        DOCKER_TTY=-it oven_run /bin/bash ;;
    ""|-h|--help)
        sed -n '2,/^set -e/{/^set -e/!p}' "$0" | sed 's/^# \{0,1\}//' ;;
    *)
        echo "unknown stage: $1  (build | package | shell | save-image)" >&2
        exit 1 ;;
esac
