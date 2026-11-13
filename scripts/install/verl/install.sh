#!/usr/bin/env bash
# The training and rollout environment, $WORK_DIR/envs/verl: torch and vLLM on CUDA 13, verl from its patched checkout
# (3rdparty/verl) and PEAR, both editable, and what is compiled for a GPU: the CUDA toolkit inside the environment,
# FlashInfer's precompiled kernels, flash-attn and the patched LMCache wheel (3rdparty/LMCache).
#   WORK_DIR=<work dir> bash scripts/install.sh verl
#   NO_GPU=1: without what is compiled for a GPU, which is what the CPU tests need
#   CUDA_ARCH=<compute capability>: the GPU to compile for (10.3 a GB300, 9.0 an H100) when it is not the machine's own
#   MAX_JOBS=<n>: the parallel compile jobs of every source build
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

VERL_SRC=$ROOT/3rdparty/verl/src
VERL_ENV=$ENVS/verl
PYTHON=$VERL_ENV/bin/python
LMCACHE_DIR=$ROOT/3rdparty/LMCache
# the parallel compile jobs of each source build on a machine with as many CPUs, unless MAX_JOBS names another number
FLASH_ATTN_2_JOB_N=64
FLASH_ATTN_3_JOB_N=8
LMCACHE_JOB_N=32
# nvcc's own threads in the flash-attn 3 build
FLASH_ATTN_3_NVCC_THREAD_N=4
# the flash-attn 3 build without what Qwen3.5's full attention does not use: it keeps bf16, head dimension 256, varlen
# and SM90
FLASH_ATTN_3_SWITCHES=(
  FLASH_ATTENTION_FORCE_BUILD=TRUE
  FLASH_ATTENTION_DISABLE_FP16=TRUE
  FLASH_ATTENTION_DISABLE_FP8=TRUE
  FLASH_ATTENTION_DISABLE_SM80=TRUE
  FLASH_ATTENTION_DISABLE_HDIM64=TRUE
  FLASH_ATTENTION_DISABLE_HDIM96=TRUE
  FLASH_ATTENTION_DISABLE_HDIM128=TRUE
  FLASH_ATTENTION_DISABLE_HDIM192=TRUE
  FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
  FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
  FLASH_ATTENTION_DISABLE_SPLIT=TRUE
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
  FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
  FLASH_ATTENTION_DISABLE_LOCAL=TRUE
  FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
)

# for every install below: the versions PEAR runs with, each with the hashes of its files, and the versions that
# replace what another package requires
export UV_CONSTRAINT=$INSTALL_DIR/verl/constraints.txt
export UV_OVERRIDE=$INSTALL_DIR/verl/overrides.txt

# The parallel compile jobs of a source build whose own number is $1.
compile_job_n() {
  local cpu_n

  cpu_n=$(nproc)
  echo "${MAX_JOBS:-$((cpu_n < $1 ? cpu_n : $1))}"
}

# The environment's packages in one resolution: the GPU stack and what PEAR adds (the extra verl of pyproject.toml) with
# the requirements of verl, which is checked out at its pinned commit with the patches of 3rdparty/verl when it is
# missing. Then verl and PEAR, editable: vLLM finds PEAR's plugin (pyproject.toml) through the environment's metadata.
install_packages() {
  create_environment "$VERL_ENV" "$VERL_PYTHON_VERSION"
  bash "$ROOT/3rdparty/verl/scripts/checkout_verl.sh"

  uv pip install --python "$PYTHON" --torch-backend="$CUDA_TAG" \
    -r "$ROOT/pyproject.toml" --extra verl -r "$VERL_SRC/requirements.txt"
  uv pip install --python "$PYTHON" --no-deps -e "$VERL_SRC" -e "$ROOT"
}

# The compiler, nvvm and the remaining headers next to the library wheels torch brought, which makes the environment a
# complete CUDA_HOME for FlashInfer's JIT, torch extensions and the builds below: a machine needs no CUDA toolkit of its
# own. With them (the extra verl_gpu) FlashInfer's precompiled kernels, from its own index, and the builds' tools.
install_cuda_toolkit() {
  uv pip install --python "$PYTHON" --find-links "$FLASHINFER_CUBIN_WHEELS_URL" \
    -r "$ROOT/pyproject.toml" --extra verl_gpu
  link_cuda_toolkit "$CUDA_TOOLKIT"
}

# flash-attn 2 for the FSDP actor (removing padding and Ulysses sequence parallelism need its varlen kernels). It has no
# wheel for this torch on CUDA 13, so it is built for CUDA_ARCH, in the environment itself (no isolation).
install_flash_attn_2() {
  local inputs

  inputs=$(cat <(grep -E '^(flash-attn|torch)==' "$UV_CONSTRAINT") <(echo "$CUDA_ARCH $VERL_PYTHON_VERSION") |
    inputs_hash)
  if is_made_from "$VERL_ENV/.pear_inputs_flash_attn_2" "$inputs"; then
    return
  fi

  # FLASH_ATTN_CUDA_ARCHS: flash-attn adds a -gencode flag for 80, 90, 100 and 120 only, and torch adds the one of
  # TORCH_CUDA_ARCH_LIST to a build that names none: a GB300's 103 keeps flash-attn's own four targets out
  CUDA_HOME=$CUDA_TOOLKIT \
    TORCH_CUDA_ARCH_LIST=$CUDA_ARCH \
    FLASH_ATTN_CUDA_ARCHS=${CUDA_ARCH//./} \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS=$(compile_job_n "$FLASH_ATTN_2_JOB_N") \
    uv pip install --python "$PYTHON" --no-build-isolation --reinstall-package flash-attn \
    --refresh-package flash-attn flash-attn
  echo "$inputs" > "$VERL_ENV/.pear_inputs_flash_attn_2"
}

# flash-attn 3 (Hopper) for the H100 actor's language model, built from its pinned commit; the wheel stays in
# $WORK_DIR/wheels under the hash of what it was built from.
install_flash_attn_3() {
  local checkout=$WORK_DIR/src/flash-attention
  local inputs wheel_dir wheels

  inputs=$(cat <(grep '^torch==' "$UV_CONSTRAINT") \
    <(echo "$FLASH_ATTN_3_COMMIT $VERL_PYTHON_VERSION ${FLASH_ATTN_3_SWITCHES[*]}") | inputs_hash)
  wheel_dir=$WORK_DIR/wheels/flash_attn_3-$inputs
  wheels=("$wheel_dir"/flash_attn_3-*.whl)
  if [ ! -e "${wheels[0]}" ]; then
    if [ ! -d "$checkout/.git" ]; then
      git clone -q "$FLASH_ATTN_3_URL" "$checkout"
    fi
    git -C "$checkout" fetch -q origin "$FLASH_ATTN_3_COMMIT"
    git -C "$checkout" checkout -q "$FLASH_ATTN_3_COMMIT"
    git -C "$checkout" submodule update -q --init --depth 1 csrc/cutlass
    (
      cd "$checkout/hopper"
      env CUDA_HOME="$CUDA_TOOLKIT" \
        MAX_JOBS="$(compile_job_n "$FLASH_ATTN_3_JOB_N")" \
        NVCC_THREADS="$FLASH_ATTN_3_NVCC_THREAD_N" \
        "${FLASH_ATTN_3_SWITCHES[@]}" \
        "$PYTHON" setup.py -q bdist_wheel --dist-dir "$wheel_dir"
    )
    wheels=("$wheel_dir"/flash_attn_3-*.whl)
  fi

  if ! is_made_from "$VERL_ENV/.pear_inputs_flash_attn_3" "$inputs"; then
    uv pip install --python "$PYTHON" --no-deps --reinstall-package flash-attn-3 "${wheels[0]}"
    echo "$inputs" > "$VERL_ENV/.pear_inputs_flash_attn_3"
  fi
}

# The patched LMCache wheel with its requirements, built for CUDA_ARCH from the checkout of $WORK_DIR/src/LMCache,
# which 3rdparty/LMCache's scripts make and compare with the patches (a checkout that differs ends the build); the
# wheel stays in $WORK_DIR/wheels under the hash of what it was built from.
install_lmcache() {
  local inputs wheel_dir wheels

  inputs=$(cat "$LMCACHE_DIR/UPSTREAM" "$LMCACHE_DIR"/patches/* "$LMCACHE_DIR"/scripts/*.sh \
    <(grep '^torch==' "$UV_CONSTRAINT") <(echo "$CUDA_ARCH $VERL_PYTHON_VERSION") | inputs_hash)
  wheel_dir=$WORK_DIR/wheels/lmcache-$inputs
  wheels=("$wheel_dir"/lmcache-*.whl)
  if [ ! -e "${wheels[0]}" ]; then
    LMCACHE_SRC=$WORK_DIR/src/LMCache \
      PYTHON=$PYTHON \
      CUDA_HOME=$CUDA_TOOLKIT \
      TORCH_CUDA_ARCH_LIST=$CUDA_ARCH \
      MAX_JOBS=$(compile_job_n "$LMCACHE_JOB_N") \
      bash "$LMCACHE_DIR/scripts/build_lmcache_wheel.sh" "$wheel_dir"
    wheels=("$wheel_dir"/lmcache-*.whl)
  fi

  if ! is_made_from "$VERL_ENV/.pear_inputs_lmcache" "$inputs"; then
    uv pip install --python "$PYTHON" --reinstall-package lmcache "${wheels[0]}"
    echo "$inputs" > "$VERL_ENV/.pear_inputs_lmcache"
  fi
}

# CUDA_ARCH: the compute capability the source builds target: the caller's, else the machine's first GPU's, else
# Blackwell's. A machine with the driver's tools and no GPU prints its error on stdout, so only a compute capability
# is taken from the output.
MACHINE_CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null |
  grep -m 1 -xE '[0-9]+\.[0-9]+' || true)
CUDA_ARCH=${CUDA_ARCH:-${MACHINE_CUDA_ARCH:-$BLACKWELL_CUDA_ARCH}}

require_tools curl getent tar sha256sum git perl
install_uv
install_packages
if [ -z "${NO_GPU:-}" ]; then
  require_tools gcc g++
  # where the environment's CUDA wheels put the toolkit, among the packages of its Python
  CUDA_TOOLKIT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
  install_cuda_toolkit
  install_flash_attn_2
  # flash-attn 4 (CuTe DSL kernels, Blackwell) for the actor. Without its dependencies: its apache-tvm-ffi requirement
  # conflicts with vLLM's, and the CuTe DSL comes at the version of overrides.txt. Its pinned version is a pre-release.
  uv pip install --python "$PYTHON" --no-deps --prerelease=allow flash-attn-4
  if [ "$CUDA_ARCH" = "$HOPPER_CUDA_ARCH" ]; then
    install_flash_attn_3
  fi
  install_lmcache
fi
# The environment's test, under the variables its processes run with: FlashInfer makes its workspace when it is
# imported, under the home directory unless they name another place.
(
  export VERL_ENV LOCAL_DIR=$WORK_DIR/cache/environment_test
  # shellcheck source=../../gpu_process_variables.sh
  source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
  "$PYTHON" -m pytest "$ROOT/tests/test_training_environment_on_cpu.py"
)
