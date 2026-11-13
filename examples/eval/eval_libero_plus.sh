#!/usr/bin/env bash
# Evaluates a model on LIBERO-plus's perturbed tasks (all, 10030 episodes) without training (pear/eval/embodied.py):
# the training rollout on the episodes, every finished group recorded instead of trained on, every GPU machine serving
# vLLM replicas. Records and a summary land in RUN/val_dumps/eval_<EVAL_STEP>{.jsonl,_summary.json}. Run it on the Ray
# head of a cluster started with scripts/start_ray_node.sh, every GPU machine as rollout (examples/README.md). Extra
# arguments are Hydra overrides:
#   WORK_DIR=<work dir> GPU=<gb300 | h100> ROLLOUT_NNODES=<rollout machines> MODEL=<model dir> \
#     EVAL_STEP=<an integer, the checkpoint's step> RUN=<run dir> bash examples/eval/eval_libero_plus.sh
set -euo pipefail

# resolved: the simulator machines' containers mount the repository at its resolved path
ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "$ROOT"

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
GPU=${GPU:?gb300 | h100}
RUN=${RUN:?the run directory}
ROLLOUT_NNODES=${ROLLOUT_NNODES:?the GPU machines of the Ray cluster, all started as rollout}
MODEL=${MODEL:?the model directory to evaluate}
# what a recipe knows of the model: its file of examples/models
MODEL_FILE=${MODEL_FILE:-$ROOT/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh}
EVAL_STEP=${EVAL_STEP:?an integer: the step of the evaluated checkpoint}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
# the head's own disk: the LOCAL_DIR scripts/start_ray_node.sh got there, with the caches of what is compiled
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}
# the episodes: files of pear/envs/libero_plus/configs or your own, comma-separated
EPISODES=${EPISODES:-$ROOT/pear/envs/libero_plus/configs/all.json}
# trajectories per episode
N=${N:-1}
# token budgets; vLLM's max_model_len is their sum, at most the model's context
MAX_PROMPT=${MAX_PROMPT:-4096}
MAX_RESPONSE=${MAX_RESPONSE:-256000}
# add "wandb" for Weights & Biases (WANDB_API_KEY in the machines' environment)
LOGGER=${LOGGER:-'["console","file","tensorboard"]'}
# ---- end user-adjustable ----

# the run's longest sequence
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))
# the model on this GPU: its arguments, its sizes, its answer parser and its image tokens
# shellcheck source=../models/qwen3_8_27b_pear/qwen3_8_27b.sh
source "$MODEL_FILE"

# what the Ray workers read, next to the variables of pear/configs/base.yaml
ENV_VAR=+ray_kwargs.ray_init.runtime_env.env_vars

# ---- environment family ----
# LIBERO-plus: its servers on the rollout machines, one per CPU slot, each a simulator process in its environment,
# rendering on the CPU
TOOL_CONFIG=$ROOT/pear/envs/libero_plus/configs/tool_config.yaml
# one 512 x 512 LIBERO-plus frame in the model's tokens
IMAGE_TOKENS=$(model_image_tokens 512 512)
FAMILY_ARGS=(
  "$ENV_VAR.LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero_plus/etc/libero"
  "$ENV_VAR.LIBERO_PYTHON=$WORK_DIR/envs/libero_plus/bin/python"
  "$ENV_VAR.LIBERO_MESA_DIR=$WORK_DIR/envs/mesa"
  # LIBERO-plus's sensor noise runs ImageMagick
  "$ENV_VAR.MAGICK_HOME='$WORK_DIR/envs/imagemagick'"
)

# ---- derived ----
# images in one prompt: only the token budget bounds the observations
LIMIT_IMAGES=$((MAX_MODEL_LEN / IMAGE_TOKENS))
VERL_SRC=$ROOT/3rdparty/verl/src
# the run directory, resolved: every machine's workers write there
RUN_DIR=$(readlink -m "$RUN")
PEAR_LOG_DIR=$RUN_DIR/sim
START_TIME=$(date -u +%Y%m%dT%H%M%SZ)

# ---- parameter arrays ----
DATA_ARGS=(
  # the episodes go through the training rollout: its dataloader, rollout.n samples each, its session cap and refill
  "data.train_files=[$EPISODES]"
  "data.val_files=[$EPISODES]"
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESPONSE"
)

ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES"
  actor_rollout_ref.rollout.n_gpus_per_node="$GPUS_PER_NODE"
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  +actor_rollout_ref.rollout.limit_images="$LIMIT_IMAGES"
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
)

TRAINER_ARGS=(
  # the run directory: the config takes the workers' paths from it
  run_dir="$RUN_DIR"
  trainer.eval_step="$EVAL_STEP"
  trainer.validation_data_dir="$RUN/val_dumps"
  trainer.logger="$LOGGER"
  trainer.experiment_name="$(basename "$RUN")"
  # Hydra's record of a launch (composed config, overrides) goes to the run, not to the working directory
  hydra.run.dir="$RUN/hydra/$START_TIME"
  # sessions a replica keeps generating, refilled as episodes finish; the router pins each to a replica. The replicas'
  # decoding slots, the agent loop's processes and the run's session cap follow from it (run_sizes of base.yaml)
  run_sizes.sessions_per_replica="$SESSIONS_PER_REPLICA"
)

# the modules verl imports in every process, the driver's included: they register PEAR's vLLM replica under
# rollout.name=vllm and the model's answer parser under rollout.multi_turn.format
EXTERNAL_MODULES=pear.rollout.server,$MODEL_MODULES
RAY_ARGS=(
  "$ENV_VAR.VERL_USE_EXTERNAL_MODULES='$EXTERNAL_MODULES'"
  # this tree's pear and verl; pear/utils/sitecustomize presizes each process's descriptor table
  "$ENV_VAR.PYTHONPATH='$ROOT/pear/utils/sitecustomize:$ROOT:$VERL_SRC'"
  "$ENV_VAR.VERL_FILE_LOGGER_PATH=$RUN/metrics_$START_TIME.jsonl"
  "${FAMILY_ARGS[@]}"
)

# ---- launch ----
mkdir -p "$RUN"
# the environment of a GPU machine's processes for the one this recipe starts on the head, which imports vLLM and
# Triton: they compile into the machine's caches, as in a Ray worker
# shellcheck source=../../scripts/gpu_process_variables.sh
source "$ROOT/scripts/gpu_process_variables.sh" "$VERL_ENV"
export RAY_ADDRESS=${RAY_ADDRESS:-auto}
export PEAR_LOG_DIR
export WANDB_DIR=$RUN
export VERL_USE_EXTERNAL_MODULES=$EXTERNAL_MODULES
export PYTHONPATH=$ROOT/pear/utils/sitecustomize:$ROOT:$VERL_SRC${PYTHONPATH:+:$PYTHONPATH}
# verl: the launch goes on only with the checkout scripts/install.sh verl built, the upstream commit of
# 3rdparty/verl/UPSTREAM with 3rdparty/verl/patches
bash "$ROOT/3rdparty/verl/scripts/checkout_verl.sh" --check | tee -a "$RUN/train.log"
# next to the driver, the progress panel's non-interactive mode: a line per refresh into RUN/progress.log, the numbers
# in RUN/progress.json
python -m pear.monitor.progress_panel \
  --target "$N" \
  --group "$N" \
  --run "$RUN" \
  --json \
  >> "$RUN/progress.log" 2>&1 &
PROGRESS_PID=$!
# it may have ended already
trap 'kill "$PROGRESS_PID" 2> /dev/null || true' EXIT
# appended: a resumed run keeps one log
python -m pear.eval.embodied \
  "${DATA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${MODEL_ROLLOUT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${TRAINER_ARGS[@]}" \
  "${RAY_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
