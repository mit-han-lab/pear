#!/usr/bin/env bash
# Builds the LMCache wheel of the training and rollout environment from the checkout LMCACHE_SRC
# (checkout_lmcache.sh).
#   PYTHON=<environment>/bin/python LMCACHE_SRC=<directory> CUDA_HOME=<toolkit> \
#     TORCH_CUDA_ARCH_LIST=<compute capability> MAX_JOBS=<n> \
#     bash 3rdparty/LMCache/scripts/build_lmcache_wheel.sh <wheel directory>
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../UPSTREAM
source "$PROJECT_DIR/UPSTREAM"
WHEEL_DIR=${1:?the directory the wheel goes to}
PYTHON=${PYTHON:?the Python of the training and rollout environment}
export LMCACHE_SRC=${LMCACHE_SRC:?the directory of the LMCache checkout}
export CUDA_HOME=${CUDA_HOME:?the CUDA toolkit to compile with}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:?the compute capability to compile for}
export MAX_JOBS=${MAX_JOBS:?the parallel compile jobs}
export PATH=$CUDA_HOME/bin:$PATH
# torch's CUDA headers include those of the CUDA libraries (cusparse.h, cublas_v2.h), which the toolkit's include
# directory carries. They go after nvcc's own: that directory also holds a crt/ of the toolkit's nvcc version, and a
# crt of another version does not compile.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -I$CUDA_HOME/include"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"

bash "$PROJECT_DIR/scripts/checkout_lmcache.sh"
mkdir -p "$WHEEL_DIR"
# a uv environment has no pip, and the build runs through pip wheel
uv pip install -q --python "$PYTHON" pip setuptools_scm ninja
cd "$LMCACHE_SRC"
SETUPTOOLS_SCM_PRETEND_VERSION=$LMCACHE_VERSION "$PYTHON" -m pip wheel . --no-deps --no-build-isolation -w "$WHEEL_DIR"
ls -la "$WHEEL_DIR"/lmcache-*.whl
