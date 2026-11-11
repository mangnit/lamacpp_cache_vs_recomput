#!/usr/bin/env bash
# Builds the four llama.cpp versions compared here, each with the probes, from GitHub source tarballs.
#
#   ./build.sh                      # WORK=./work  CUDA_ARCH=75  JOBS=4
#   CUDA_ARCH=86 JOBS=8 ./build.sh  # another GPU
#
# Needs CUDA (nvcc), cmake >= 3.18, curl, python3. Each build takes a while: llama.cpp compiles every CUDA
# kernel. Executables land in $WORK/build-<A|B|C|D>/ (llama-bench in .../bin/).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$ROOT/work}; ARCH=${CUDA_ARCH:-75}; JOBS=${JOBS:-4}
A=740251a920abf92b7744d9c0518ca157db0db579   # parent of #31247 ("before")
B=8fc0376721e9662d498a5706d7ee8e7574ce99f4   # #31247 merged ("after")
D=cd4ed902e174e54d7198105a720bb379a526309d   # master, 2026-11-10
TARGETS="matmul_paths kv_paths ppl_paths quantize_pure llama-bench"
mkdir -p "$WORK"

fetch() {  # commit -> $WORK/src-<commit>
    [ -d "$WORK/src-$1" ] && return
    curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/$1.tar.gz" | tar -xz -C "$WORK"
    mv "$WORK/llama.cpp-$1" "$WORK/src-$1"
}

build() {  # label source-tree tag
    cmake -S "$ROOT" -B "$WORK/build-$1" -DLLAMA_CPP_DIR="$2" -DPROBE_LLAMA_TAG="$3" \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
    cmake --build "$WORK/build-$1" --target $TARGETS -j"$JOBS"
}

fetch $A; fetch $B; fetch $D
build A "$WORK/src-$A" 740251a
build B "$WORK/src-$B" 8fc0376
# C = B with the FAST_FP16 __hmul2 branch removed from the Q4_1/Q5_1 MMVQ dot products ("I9").
rm -rf "$WORK/src-C"; cp -r "$WORK/src-$B" "$WORK/src-C"
python3 "$ROOT/scripts/apply_i9_only.py" "$WORK/src-C"
build C "$WORK/src-C" 8fc0376+i9
build D "$WORK/src-$D" cd4ed90
echo "built: $WORK/build-{A,B,C,D}"
