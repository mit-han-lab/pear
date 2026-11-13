#!/usr/bin/env bash
# Serves Qwen3.8-27B to one session at a time on one GPU of this machine (pear/deploy/server/main.py), in this terminal
# until it is stopped: vLLM's OpenAI-compatible server in the server's environment (scripts/install.sh server), with
# the Qwen3.5 family's deploy kernels, which pick the decode megakernel of the GPU and the checkpoint. The GPU's engine
# arguments are gpu/<GPU>.yaml; extra arguments are Hydra overrides (kernels=null: vLLM's own kernels):
#   export WORK_DIR=<work dir> GPU=<gb300 | rtx5090> MODEL=<checkpoint> DRAFT=<DFlash draft> RUN=<run dir>
#   bash examples/models/qwen3_8_27b_pear/deploy/run_server.sh
set -euo pipefail

RECIPE_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$RECIPE_DIR/../../../.." && pwd)

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
# the GPU's engine arguments: a file of gpu/
GPU=${GPU:?gb300 | rtx5090}
MODEL=${MODEL:?the checkpoint}
# the DFlash draft of the model, whose verify steps the megakernels run
DRAFT=${DRAFT:?the DFlash draft}
RUN=${RUN:?the run directory}
GPU_INDEX=${GPU_INDEX:-0}
PORT=${PORT:-8000}
NAME=${NAME:-qwen3_8_27b}
SERVER_ENV=${SERVER_ENV:-$WORK_DIR/envs/server}
# the machine's own disk: the caches of what is compiled
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}
# ---- end user-adjustable ----

# shellcheck source=../../../../scripts/gpu_process_variables.sh
source "$ROOT/scripts/gpu_process_variables.sh" "$SERVER_ENV"
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
# the server starts on the CPUs next to its GPU, so what it allocates lands on their NUMA node
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$GPU_INDEX" | tr 'A-F' 'a-f')
CPUS=$(cat "/sys/bus/pci/devices/${BUS#0000}/local_cpulist")
START_TIME=$(date -u +%Y%m%dT%H%M%SZ)

exec taskset -c "$CPUS" python -m pear.deploy.server.main --config-dir "$RECIPE_DIR" +qwen3_5=deploy "+gpu=$GPU" \
  "model=$MODEL" "draft=$DRAFT" "+server={host: 0.0.0.0, port: $PORT, served_model_name: $NAME}" \
  hydra.run.dir="$RUN/hydra/$START_TIME" "$@"
