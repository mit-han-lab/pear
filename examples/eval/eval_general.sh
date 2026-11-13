#!/usr/bin/env bash
# The general-capability evaluation of one model (pear/eval/general.py, evalscope): vLLM replicas on every GPU machine,
# then a router over them and the evalscope client on one machine. The same command on the same RUN reuses the
# finished samples (evalscope's cache).
#   export WORK_DIR=<work dir> GPU=<gb300 | h100> MODEL=<model dir> NAME=<served name> RUN=<run dir>
#   bash examples/eval/eval_general.sh serve                             # on every GPU machine, left running
#   bash examples/eval/eval_general.sh run <host>:8000,<host>:8001,...   # on one of them
# run passes further arguments on to `python -m pear.eval.general run`.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
COMMAND=${1:?serve | run <replicas, host:port,...>}

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
GPU=${GPU:?gb300 | h100}
MODEL=${MODEL:?the model directory}
# what a recipe knows of the model: its file of examples/models
MODEL_FILE=${MODEL_FILE:-$ROOT/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh}
NAME=${NAME:?the name the model is served under}
RUN=${RUN:?the run directory}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
EVAL_PYTHON=${EVAL_PYTHON:-$WORK_DIR/envs/evalscope/bin/python}
# the benchmarks' datasets and what the libraries that prepare them keep, as scripts/install.sh evalscope left them
DATASET_DIR=${DATASET_DIR:-$WORK_DIR/cache/evalscope}
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear} # the machine's own disk
FIRST_REPLICA_PORT=8000
ROUTER_PORT=${ROUTER_PORT:-9000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-139264}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GPU_MEMORY_UTILIZATION=0.90
IMAGES_PER_PROMPT=64
REPLICA_POLL_S=10 # between two health checks of a replica that is still loading
ROUTER_START_S=10
# ---- end user-adjustable ----

# the model on this GPU: the GPUs of a replica and its server's own flags
# shellcheck source=../models/qwen3_8_27b_pear/qwen3_8_27b.sh
source "$MODEL_FILE"

# One replica per TP GPUs of this machine, each an OpenAI API in the model card's thinking mode, with the reasoning
# split off into reasoning_content.
serve() {
  local replica_n replica gpus
  replica_n=$(($(nvidia-smi -L | wc -l) / TP))
  # the replicas' variables, as on a machine of a Ray cluster
  # shellcheck source=../../scripts/gpu_process_variables.sh
  source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"

  REPLICA_PIDS=()
  # the replicas end with this script, as when a signal ends it; they may have ended already
  trap 'kill "${REPLICA_PIDS[@]}" 2> /dev/null || true' EXIT
  for replica in $(seq 0 $((replica_n - 1))); do
    gpus=$(seq -s, $((replica * TP)) $((replica * TP + TP - 1)))
    CUDA_VISIBLE_DEVICES=$gpus vllm serve "$MODEL" \
      --served-model-name "$NAME" \
      --host 0.0.0.0 \
      --port $((FIRST_REPLICA_PORT + replica)) \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      "${MODEL_SERVER_ARGS[@]}" \
      --enable-prefix-caching \
      --limit-mm-per-prompt "{\"image\": $IMAGES_PER_PROMPT, \"video\": 0}" \
      --generation-config vllm \
      > "$RUN/vllm_$(hostname)_$replica.log" 2>&1 &
    REPLICA_PIDS+=($!)
  done
  wait
}

# The router over the replicas, then the client from a throwaway directory on the machine's own disk: LiveCodeBench
# executes the generated programs there (evalscope's reliability guard on).
run() {
  local replicas=${1:?the replicas, host:port,host:port,...}
  shift
  local replica client_dir
  export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}

  # until every replica answers: the weights and the first compilations
  for replica in ${replicas//,/ }; do
    until curl -sf "http://$replica/health" > /dev/null; do
      sleep "$REPLICA_POLL_S"
    done
  done
  echo "$(date) replicas up: $replicas"

  "$EVAL_PYTHON" -m pear.eval.general router --port "$ROUTER_PORT" --backends "$replicas" > "$RUN/router.log" 2>&1 &
  ROUTER_PID=$!
  trap 'kill "$ROUTER_PID" 2> /dev/null || true' EXIT # it may have ended already
  sleep "$ROUTER_START_S"

  client_dir=$LOCAL_DIR/eval_general.$$
  rm -rf "$client_dir"
  mkdir -p "$client_dir"
  cd "$client_dir"
  TMPDIR=$client_dir HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1 "$EVAL_PYTHON" -m pear.eval.general run \
    --api-url "http://localhost:$ROUTER_PORT/v1" \
    --model "$NAME" \
    --out "$RUN" \
    --dataset-dir "$DATASET_DIR" \
    "$@" 2>&1 | tee -a "$RUN/eval.log"
  cd /
  rm -rf "$client_dir"
}

mkdir -p "$RUN"
case "$COMMAND" in
  serve) serve ;;
  run) run "${@:2}" ;;
  *)
    echo "unknown command $COMMAND (serve | run)"
    exit 1
    ;;
esac
