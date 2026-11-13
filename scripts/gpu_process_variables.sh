# shellcheck shell=bash
# Sourced, not run, with the Python environment the processes run in as its argument: by scripts/start_ray_node.sh,
# the recipes of examples/, tests/run_gpu_tests.sh and the installer's targets verl and libero_run. The environment of
# PEAR's processes on a GPU machine. The caller sets WORK_DIR (what scripts/install.sh built) and LOCAL_DIR (a
# directory on its own disk).
#   source scripts/gpu_process_variables.sh <an environment scripts/install.sh built>

PYTHON_ENV=${1:?the Python environment of the processes, one that scripts/install.sh built}

# the CUDA toolkit lives in the environment install.sh built, among the packages of its Python, which names their
# directory; numactl and numastat, for the LMCache sidecars, where the target numactl installed them
CUDA_HOME=$("$PYTHON_ENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
export CUDA_HOME
export PATH=$PYTHON_ENV/bin:$CUDA_HOME/bin:$WORK_DIR/envs/numactl/bin:$WORK_DIR/bin:$PATH

# JIT caches stay on the machine: Triton, Inductor and vLLM's compiled graphs reference each other by absolute path,
# and many processes compiling one kernel into a shared directory race; a machine compiles once and keeps the result
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$LOCAL_DIR/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_DIR/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$LOCAL_DIR/triton}
# the builds of torch's cpp_extension (the Qwen3.5 kernels)
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$LOCAL_DIR/torch_extensions}
# the NVIDIA driver's cache of the kernels it compiles when a process loads them, under the home directory by default
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$LOCAL_DIR/cuda}
# the cache's limit in bytes, the driver's largest: the default holds less than a trainer's kernels, which the driver
# then evicts and compiles again at every start
export CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-$((4 * 1024 * 1024 * 1024))}
# flash-attn 4's CuTe-DSL kernel cache is off by default: every trainer process would recompile all variants at
# every start, past the collective timeout
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-$LOCAL_DIR/fa4_cute}
# FlashInfer's JIT modules bake in the tvm-ffi headers of the environment that built them, and a module built by
# another tvm-ffi version aborts the worker at load: every environment keeps its own module cache
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$LOCAL_DIR/flashinfer_${PYTHON_ENV##*/}}
# FlashInfer's precompiled kernels are the environment's (flashinfer-cubin, which scripts/install.sh installs): one it
# lacks is an error of the request that launches it first, not a download from FlashInfer's artifact server
export FLASHINFER_NO_DOWNLOAD=1

# the threads of a process's OpenMP and MKL pools
GPU_PROCESS_THREAD_N=8
export OMP_NUM_THREADS=$GPU_PROCESS_THREAD_N
export MKL_NUM_THREADS=$GPU_PROCESS_THREAD_N
export LMCACHE_LOG_LEVEL=WARNING
# a run's model is a directory: the Hub's libraries ask the network for nothing, and what they keep is in the work
# directory with install.sh's downloads, not under the home directory with a login stored there. HF_HUB_OFFLINE=0
# lets them reach the Hub
export HF_HOME=${HF_HOME:-$WORK_DIR/cache/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# a tokenizer encodes on the calling thread, not on a pool of every core in each of the machine's processes
export TOKENIZERS_PARALLELISM=false
# vLLM's usage report, a post to its statistics server and a file under the home directory, is off; 0 turns it on
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
# LMCache's usage report, posts to its statistics server and a file under the home directory, is off; true turns it on
export LMCACHE_TRACK_USAGE=${LMCACHE_TRACK_USAGE:-false}
# aarch64's torch build embeds mimalloc, which returns freed memory only after a delay and only when the thread
# allocates again: a loaded checkpoint shard stays resident in every trainer unless purged at once
if [ "$(uname -m)" = aarch64 ]; then
  export MIMALLOC_PURGE_DELAY=0
fi
