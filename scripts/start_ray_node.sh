#!/usr/bin/env bash
# Starts Ray on this machine for a PEAR run and stays in the foreground until Ray stops. Run it once on every machine,
# the head first, with whatever the cluster offers (ssh, a pod's command, a batch scheduler's step).
#   export WORK_DIR=<the directory scripts/install.sh built>
#   bash scripts/start_ray_node.sh trainer                   # the first machine: a trainer machine and the Ray head
#   bash scripts/start_ray_node.sh trainer <head ip>:6379    # the other trainer machines
#   bash scripts/start_ray_node.sh rollout <head ip>:6379    # the vLLM replicas and LIBERO's simulators
#   bash scripts/start_ray_node.sh robodojo <head ip>:6379   # a RoboDojo simulator machine
#   bash scripts/start_ray_node.sh robolab <head ip>:6379    # a RoboLab simulator machine
#   NUM_CPUS=2 bash scripts/start_ray_node.sh cpu           # a Tinker coordinator, or worker with a head address
#   NUM_CPUS=8 bash scripts/start_ray_node.sh libero <head ip>:6379  # CPU-only LIBERO workers
# Start it from a clean shell: a Conda base's compiler variables and library paths would reach the simulators. On a
# shared network turn on Ray's token authentication, on every machine and for the recipe:
#   export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=<a file only you can read>
set -euo pipefail

# resolved: the container of a RoboDojo machine mounts the repository at its resolved path
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
ROLE=${1:?cpu | libero | trainer | rollout | robodojo | robolab}
HEAD_ADDRESS=${2:-} # none: this machine is the head
cd "$ROOT"

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
TINKER_ENV=${TINKER_ENV:-$WORK_DIR/envs/tinker}
# Override Ray's Python environment on simulator nodes when attaching to an existing cluster.
RAY_ENV=${RAY_ENV:-}
# the machine's own disk: not a shared file system, and not a /tmp held in memory
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}
PORT=${PORT:-6379}     # of the head
NUM_CPUS=${NUM_CPUS:-} # required allocation on CPU nodes; defaults to 120 on model nodes
# a rollout machine's LIBERO servers, one CPU slot each: a run starts one per slot of its cluster; enough that the
# machine's sessions seldom queue for a simulator, speculative decoding's shorter generation included
LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-}
# Explicit spare CPU budget on Isaac machines. Zero keeps LIBERO off those machines.
SIM_LIBERO_CPUS=${SIM_LIBERO_CPUS:-0}
# shared memory next to the replicas' LMCache tiers and TransferQueue's storage units, so not Ray's default (up to
# 30% of RAM)
OBJECT_STORE_GB=${OBJECT_STORE_GB:-}
# ---- end user-adjustable ----

# owner-only (mode 700) when this script makes it; one that exists keeps its mode
# shellcheck disable=SC2174  # the mode is for the directory named, not for the parents -p makes
mkdir -p -m 700 "$LOCAL_DIR"
export RAY_TMPDIR=${RAY_TMPDIR:-$LOCAL_DIR/ray}
GPU_N=0
RAY=${RAY_ENV:-$VERL_ENV}/bin/ray
# Ray's processes import from it: the sitecustomize.py that presizes the descriptor table, which the raylet and
# gcs_server inherit
RAY_PYTHONPATH=$ROOT/pear/utils/sitecustomize
PORT_ARGS=()
# a simulator machine's object store: the sessions' results in flight, one per App at most
SIM_OBJECT_STORE_GB=${SIM_OBJECT_STORE_GB:-8}
# Ray's workers, below the ephemeral range and Ray's own fixed ports (GCS 6379, dashboard 8265, client 10001)
MIN_WORKER_PORT=3000
MAX_WORKER_PORT=5999

# A trainer or rollout machine: the GPU processes' variables, the role as a Ray resource, the workers' ports.
configure_gpu_machine() {
  local first_ephemeral_port
  # shellcheck source=gpu_process_variables.sh
  source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
  NUM_CPUS=${NUM_CPUS:-120}
  OBJECT_STORE_GB=${OBJECT_STORE_GB:-64}
  LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-32}
  # the role, by which verl places the trainer's and the replicas' pools
  RESOURCES="{\"$ROLE\": $GPU_N}"
  if [ "$ROLE" = rollout ]; then
    RESOURCES="{\"rollout\": $GPU_N, \"libero\": $LIBERO_SERVERS_PER_NODE}"
  fi
  # Ray's workers bind ports 10002-19999 by default, which the ephemeral range can cover: a port picked with bind(0),
  # as verl's TCPStore master port, could then go to a worker that starts before the store binds it.
  read -r first_ephemeral_port _ < /proc/sys/net/ipv4/ip_local_port_range
  if [ "$first_ephemeral_port" -le "$MAX_WORKER_PORT" ]; then
    echo "ephemeral ports from $first_ephemeral_port overlap the Ray worker ports"
    exit 1
  fi
  PORT_ARGS=(--min-worker-port="$MIN_WORKER_PORT" --max-worker-port="$MAX_WORKER_PORT")
}

# A RoboDojo simulator machine, inside the Isaac container: the Apps run the image's
# Python, the machine's Ray the environment install.sh built for it, at its resolved path, which the container mounts.
configure_robodojo_machine() {
  SIM_RUNTIME_DIR=$(readlink -f "${SIM_RUNTIME_DIR:-$WORK_DIR/robodojo_runtime}")
  simulator_ray "$WORK_DIR/envs/robodojo"
  ROBODOJO_ROOT=$(readlink -f "${ROBODOJO_ROOT:-$SIM_RUNTIME_DIR/RoboDojo}")
  export ROBODOJO_ROOT
  export ROBODOJO_KIT_EXTS_DIR=$SIM_RUNTIME_DIR/kit_exts
  export ROBODOJO_LOCAL_DIR=${ROBODOJO_LOCAL_DIR:-$LOCAL_DIR/robodojo_sim}
  if [ ! -d "$ROBODOJO_KIT_EXTS_DIR" ]; then
    echo "$ROBODOJO_KIT_EXTS_DIR is missing"
    exit 1
  fi
  mkdir -p "$ROBODOJO_LOCAL_DIR"
  if [ -n "${NVIDIA_GRAPHICS_DIR:-}" ]; then
    install_graphics_libraries
  fi
  # only RoboDojo's servers ask for it (RoboDojoEnv.SERVER_RESOURCES)
  RESOURCES="{\"robodojo\": $GPU_N}"
  offer_simulator_resources
}

# A driver without its graphics libraries: those of its version (scripts/install.sh robodojo), mounted at
# NVIDIA_GRAPHICS_DIR, go into the writable container. The container runtime mounts the compute libraries, some of
# them read-only (libnvidia-gpucomp): only the missing files are copied.
install_graphics_libraries() {
  local library_dir path
  library_dir=/usr/lib/$(uname -m)-linux-gnu
  for path in "$NVIDIA_GRAPHICS_DIR"/lib*; do
    if [ ! -e "$library_dir/${path##*/}" ]; then
      cp "$path" "$library_dir/"
    fi
  done
  mkdir -p /etc/vulkan/icd.d /usr/share/nvidia
  cp "$NVIDIA_GRAPHICS_DIR/nvidia_icd.json" /etc/vulkan/icd.d/
  cp "$NVIDIA_GRAPHICS_DIR/nvoptix.bin" /usr/share/nvidia/
  ldconfig
}

# A RoboLab simulator machine, without a container: the Apps run Isaac Sim's pip packages in RoboLab's uv
# environment, the machine's Ray is the training's.
configure_robolab_machine() {
  local path
  simulator_ray "$VERL_ENV"
  SIM_RUNTIME_DIR=$(readlink -f "${SIM_RUNTIME_DIR:-$WORK_DIR/robolab_runtime}")
  export ROBOLAB_PYTHON=${ROBOLAB_PYTHON:-$WORK_DIR/envs/robolab/bin/python}
  export ROBOLAB_LOCAL_DIR=${ROBOLAB_LOCAL_DIR:-$LOCAL_DIR/robolab_sim}
  # omni.client reads omniverse.toml (the local copy's aliases) in the directory OMNI_CONFIG_PATH names
  export ROBOLAB_OMNI_CONFIG_PATH=$SIM_RUNTIME_DIR/omni_config
  export ROBOLAB_HOST_LIB_DIR=$SIM_RUNTIME_DIR/host/lib
  for path in "$ROBOLAB_PYTHON" "$ROBOLAB_OMNI_CONFIG_PATH/omniverse.toml" "$ROBOLAB_HOST_LIB_DIR"; do
    if [ ! -e "$path" ]; then
      echo "$path is missing"
      exit 1
    fi
  done
  mkdir -p "$ROBOLAB_LOCAL_DIR"
  RAY_PYTHONPATH=$RAY_PYTHONPATH:$ROOT
  # only RoboLab's servers ask for it (RoboLabEnv.SERVER_RESOURCES)
  RESOURCES="{\"robolab\": $GPU_N}"
  offer_simulator_resources
}

# Keep each simulator's existing Ray environment unless the caller selects another one.
simulator_ray() {
  RAY=${RAY_ENV:-$1}/bin/ray
}

# An Isaac App's CPU and memory needs remain outside Ray's CPU scheduling. Advertise only an explicit spare budget.
offer_simulator_resources() {
  NUM_CPUS=$SIM_LIBERO_CPUS
  LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-$SIM_LIBERO_CPUS}
  check_libero_capacity
  if [ "$LIBERO_SERVERS_PER_NODE" -gt 0 ]; then
    RESOURCES="{\"$ROLE\": $GPU_N, \"libero\": $LIBERO_SERVERS_PER_NODE}"
  fi
  OBJECT_STORE_GB=${OBJECT_STORE_GB:-$SIM_OBJECT_STORE_GB}
}

check_libero_capacity() {
  if ! [[ $NUM_CPUS =~ ^[0-9]+$ && $LIBERO_SERVERS_PER_NODE =~ ^[0-9]+$ ]] ||
    [ "$LIBERO_SERVERS_PER_NODE" -gt "$NUM_CPUS" ]; then
    echo "LIBERO_SERVERS_PER_NODE must fit the offered CPU slots (SIM_LIBERO_CPUS on Isaac machines)" >&2
    exit 1
  fi
}

configure_cpu_machine() {
  RAY=${RAY_ENV:-$TINKER_ENV}/bin/ray
  : "${NUM_CPUS:?set NUM_CPUS to the CPU slots allocated to this Ray node}"
  OBJECT_STORE_GB=${OBJECT_STORE_GB:-2}
  if [ "$ROLE" = libero ]; then
    LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-$NUM_CPUS}
  else
    LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-0}
  fi
  check_libero_capacity
  RESOURCES="{\"libero\": $LIBERO_SERVERS_PER_NODE}"
  RAY_PYTHONPATH=$RAY_PYTHONPATH:$ROOT
}

# A limit on a machine's threads may apply: without these, numpy's OpenBLAS alone starts one thread per core in each
# of the machine's processes. The Apps set their own values.
configure_simulator_processes() {
  export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
}

case "$ROLE" in
  cpu | libero)
    configure_cpu_machine
    configure_simulator_processes
    ;;
  trainer | rollout)
    GPU_N=$(nvidia-smi -L | wc -l)
    configure_gpu_machine
    ;;
  robodojo)
    GPU_N=$(nvidia-smi -L | wc -l)
    configure_robodojo_machine
    configure_simulator_processes
    ;;
  robolab)
    GPU_N=$(nvidia-smi -L | wc -l)
    configure_robolab_machine
    configure_simulator_processes
    ;;
  *)
    echo "unknown role $ROLE (cpu | libero | trainer | rollout | robodojo | robolab)"
    exit 1
    ;;
esac

if [ "${LIBERO_SERVERS_PER_NODE:-0}" -gt 0 ]; then
  export LIBERO_PYTHON=${LIBERO_PYTHON:-$WORK_DIR/envs/libero/bin/python}
  export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$WORK_DIR/envs/libero/etc/libero}
  export LIBERO_MESA_DIR=${LIBERO_MESA_DIR:-$WORK_DIR/envs/mesa}
fi
export USER=${USER:-$(id -un)}

# No dashboard and no Ray Client on the head: `ray start --head` waits at most 30 s for the raylet, which starts only
# after the dashboard's subprocess modules import; from a cold page cache on a shared file system that takes longer.
CONNECT_ARGS=(--head --port="$PORT" --include-dashboard=false)
if [ -n "$HEAD_ADDRESS" ]; then
  CONNECT_ARGS=(--address="$HEAD_ADDRESS")
fi
mkdir -p "$RAY_TMPDIR"
PYTHONPATH=$RAY_PYTHONPATH exec "$RAY" start --block --disable-usage-stats \
  --num-cpus="$NUM_CPUS" \
  --num-gpus="$GPU_N" \
  --resources="$RESOURCES" \
  --temp-dir="$RAY_TMPDIR" \
  --object-store-memory=$((OBJECT_STORE_GB << 30)) \
  "${PORT_ARGS[@]}" \
  "${CONNECT_ARGS[@]}"
