#!/usr/bin/env bash
# The training and rollout environment, $WORK_DIR/envs/verl: torch and vLLM on CUDA 13, verl from its patched checkout
# (3rdparty/verl) and PEAR, both editable, and what is compiled for a GPU: the CUDA toolkit inside the environment,
# FlashInfer's precompiled kernels, flash-attn and the patched LMCache wheel (3rdparty/LMCache).
#   WORK_DIR=<work dir> bash scripts/install.sh verl
#   NO_GPU=1: without what is compiled for a GPU, which is what the CPU tests need
#   CUDA_ARCH=<compute capability>: the GPU to compile for (10.3 a GB300, 9.0 an H100) when it is not the machine's own
#   MAX_JOBS=<n>: the parallel compile jobs of every source build
#   MEMORY_GIB=<n>: the memory the source builds may take, when less than the machine and its cgroup leave them
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

VERL_SRC=$ROOT/3rdparty/verl/src
FLASH_ATTENTION_SRC=$WORK_DIR/src/flash-attention
VERL_ENV=$ENVS/verl
PYTHON=$VERL_ENV/bin/python
LMCACHE_DIR=$ROOT/3rdparty/LMCache
# the parallel compile jobs of each source build on a machine with as many CPUs and the memory for them, unless MAX_JOBS
# names another number
FLASH_ATTN_2_JOB_N=64
FLASH_ATTN_3_JOB_N=8
LMCACHE_JOB_N=32
# the memory one compile job of each source build takes at its peak, in GiB
FLASH_ATTN_2_JOB_GIB=8
FLASH_ATTN_3_JOB_GIB=8
LMCACHE_JOB_GIB=4
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

# The parallel compile jobs of a source build whose own number is $1 and one of whose jobs takes up to $2 GiB: MAX_JOBS
# when the caller names it, else no more than that number, the machine's CPUs and the jobs the build memory holds.
compile_job_n() {
  local job_n=$1 job_gib=$2
  local cpu_n memory_job_n

  cpu_n=$(nproc)
  memory_job_n=$(($(build_memory_kib) / (job_gib * 1024 * 1024)))
  job_n=$((cpu_n < job_n ? cpu_n : job_n))
  job_n=$((memory_job_n < job_n ? memory_job_n : job_n))
  echo "${MAX_JOBS:-$((job_n > 0 ? job_n : 1))}"
}

# The memory the source builds may take, in KiB: MEMORY_GIB when the caller names it, else what the machine has
# available, or less where a cgroup v2 that holds this process (a Slurm job's, a container's) has less left under its
# limit.
build_memory_kib() {
  local memory_kib directory limit inactive_file working_set

  if [ -n "${MEMORY_GIB:-}" ]; then
    echo $((MEMORY_GIB * 1024 * 1024))
    return
  fi

  memory_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  directory=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  directory=${directory%/}
  while true; do
    limit=$(cat "$directory/memory.max" 2> /dev/null || echo max)
    if [ "$limit" != max ]; then
      # what the cgroup holds less its inactive page cache (the kubelet's working set): the kernel frees that cache
      # first, so a cgroup full of cached files still has room for a build
      inactive_file=$(awk '/^inactive_file / {print $2}' "$directory/memory.stat")
      working_set=$(($(cat "$directory/memory.current") - inactive_file))
      limit=$(((limit - working_set) / 1024))
      memory_kib=$((limit < memory_kib ? limit : memory_kib))
    fi
    if [ "$directory" = /sys/fs/cgroup ]; then
      break
    fi
    directory=$(dirname "$directory")
  done
  echo "$memory_kib"
}

# The flash-attention checkout at its pinned commit, with the CUTLASS that flash-attn 2 and 3 compile against.
checkout_flash_attention() {
  if [ ! -d "$FLASH_ATTENTION_SRC/.git" ]; then
    git clone -q "$FLASH_ATTN_URL" "$FLASH_ATTENTION_SRC"
  fi
  git -C "$FLASH_ATTENTION_SRC" fetch -q origin "$FLASH_ATTN_COMMIT"
  git -C "$FLASH_ATTENTION_SRC" checkout -q "$FLASH_ATTN_COMMIT"
  git -C "$FLASH_ATTENTION_SRC" submodule update -q --init --depth 1 csrc/cutlass
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
  # FlashInfer's prebuilt JIT modules of a GB300 (the extra verl_jit_cache), without the package of every GPU
  # architecture that flashinfer-jit-cache requires, as FlashInfer's own command installs them.
  # x86_64 machines keep FlashInfer's build at first use: its sm90a package lacks trtllm_mnnvl_comm.
  uv pip install --python "$PYTHON" --no-deps --index "$FLASHINFER_JIT_CACHE_INDEX_URL" \
    -r "$ROOT/pyproject.toml" --extra verl_jit_cache
  link_cuda_toolkit "$CUDA_TOOLKIT"
}

# flash-attn 2 for the FSDP actor (removing padding and Ulysses sequence parallelism need its varlen kernels), built
# for CUDA_ARCH from the flash-attention checkout, in the environment itself (no isolation).
install_flash_attn_2() {
  local inputs

  inputs=$(cat <(grep '^torch==' "$UV_CONSTRAINT") <(echo "$FLASH_ATTN_COMMIT $CUDA_ARCH $VERL_PYTHON_VERSION") |
    inputs_hash)
  if is_made_from "$VERL_ENV/.pear_inputs_flash_attn_2" "$inputs"; then
    return
  fi

  checkout_flash_attention
  # FLASH_ATTN_CUDA_ARCHS empty: flash-attn adds no -gencode flag (unset, its five targets; a GB300's 103, PTX alone),
  # and torch adds TORCH_CUDA_ARCH_LIST's compiled target to a build that names none
  CUDA_HOME=$CUDA_TOOLKIT \
    TORCH_CUDA_ARCH_LIST=$CUDA_ARCH \
    FLASH_ATTN_CUDA_ARCHS='' \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    MAX_JOBS=$(compile_job_n "$FLASH_ATTN_2_JOB_N" "$FLASH_ATTN_2_JOB_GIB") \
    uv pip install --python "$PYTHON" --no-build-isolation --reinstall-package flash-attn \
    --refresh-package flash-attn "$FLASH_ATTENTION_SRC"
  echo "$inputs" > "$VERL_ENV/.pear_inputs_flash_attn_2"
}

# flash-attn 3 (Hopper) for the H100 actor's language model, built from the flash-attention checkout; the wheel stays
# in $WORK_DIR/wheels under the hash of what it was built from.
install_flash_attn_3() {
  local inputs wheel_dir wheels

  inputs=$(cat <(grep '^torch==' "$UV_CONSTRAINT") \
    <(echo "$FLASH_ATTN_COMMIT $VERL_PYTHON_VERSION ${FLASH_ATTN_3_SWITCHES[*]}") | inputs_hash)
  wheel_dir=$WORK_DIR/wheels/flash_attn_3-$inputs
  wheels=("$wheel_dir"/flash_attn_3-*.whl)
  if [ ! -e "${wheels[0]}" ]; then
    checkout_flash_attention
    (
      cd "$FLASH_ATTENTION_SRC/hopper"
      env CUDA_HOME="$CUDA_TOOLKIT" \
        MAX_JOBS="$(compile_job_n "$FLASH_ATTN_3_JOB_N" "$FLASH_ATTN_3_JOB_GIB")" \
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
      MAX_JOBS=$(compile_job_n "$LMCACHE_JOB_N" "$LMCACHE_JOB_GIB") \
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
