#!/usr/bin/env bash
# Stage 3 of Qwen3.8-27B's training, from a model directory written from a checkpoint of stage 2
# (scripts/export_checkpoint.sh): GRPO on LIBERO's four suites (libero_all, 2000 episodes) and RoboDojo's pickup_align
# (328 episodes) as one dataset, which the DAPO filter samples as it finds signal, at the 256k budget with token-mean
# (RoboDojo's episodes stop at 160k tokens), 128 groups x 8 per step: the actor on the trainer machines, standalone
# vLLM replicas (verl's separate_async with hybrid_engine=False) and LIBERO's simulators on the rollout machines,
# RoboDojo's on machines of their own. Run it on the Ray head of a cluster started with scripts/start_ray_node.sh
# (examples/README.md). Extra arguments are Hydra overrides; an existing RUN resumes from RUN/ckpt:
#   WORK_DIR=<work dir> GPU=<gb300 | h100> TRAINER_NNODES=<trainer machines> ROLLOUT_NNODES=<rollout machines> \
#     SIM_NNODES=<RoboDojo machines> MODEL=<stage 2's model directory> RUN=<run dir> \
#     bash examples/models/qwen3_8_27b_pear/train/run_stage3.sh
set -euo pipefail

# resolved: the simulator machines' containers mount the repository at its resolved path
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)
cd "$ROOT"

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
GPU=${GPU:?gb300 | h100}
RUN=${RUN:?the run directory}
TRAINER_NNODES=${TRAINER_NNODES:?the machines started as trainer}
ROLLOUT_NNODES=${ROLLOUT_NNODES:?the machines started as rollout}
SIM_NNODES=${SIM_NNODES:?the RoboDojo simulator machines}
MODEL=${MODEL:?the model directory written from a checkpoint of stage 2}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
# the head's own disk: the LOCAL_DIR scripts/start_ray_node.sh got there, with the caches of what is compiled
LOCAL_DIR=${LOCAL_DIR:-/var/tmp/${USER:-$(id -un)}/pear}
# upper bounds: a run ends when you stop it
STEPS=${STEPS:-1000}
EPOCHS=${EPOCHS:-100}
# 128 usable groups x 8 = 1024 trajectories per step. Use a power-of-two number of trainer machines: the data-parallel
# size (trainer GPUs / SP) then divides the batch, which verl v1 would otherwise pad to lcm(dp, 8 x BATCH)
BATCH=${BATCH:-128}
N=${N:-8}
# token budgets; vLLM's max_model_len is their sum, at most the model's context
MAX_PROMPT=${MAX_PROMPT:-4096}
MAX_RESPONSE=${MAX_RESPONSE:-256000}
# the Apps of a RoboDojo server, an episode each; unset: the tool config's own number, an H100's (a GB300 holds 24)
ROBODOJO_APPS_PER_GPU=${ROBODOJO_APPS_PER_GPU:-}
# pickup_align's episodes stop at 160k tokens, LIBERO's at MAX_RESPONSE
export ROBODOJO_RESPONSE_BUDGET=${ROBODOJO_RESPONSE_BUDGET:-160000}
# add "wandb" for Weights & Biases (WANDB_API_KEY in the machines' environment)
LOGGER=${LOGGER:-'["console","file","tensorboard"]'}
# ---- end user-adjustable ----

# the run's longest sequence
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))
# the model on this GPU: its arguments, its sizes, its answer parser and its image tokens
# shellcheck source=../qwen3_8_27b.sh
source "$ROOT/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh"

# ---- derived ----
# the run's RoboDojo servers, a GPU of a simulator machine each, whose Apps, an episode each, cap RoboDojo's sessions
# in flight
ROBODOJO_SERVER_N=$((SIM_NNODES * GPUS_PER_NODE))
# pickup_align's 328 rows this many times next to libero_all's 2000: the dataloader draws about one of each
PICKUP_ALIGN_REPEAT_N=6
EPISODES=$ROOT/pear/envs/libero/configs/libero_all.json
for _ in $(seq "$PICKUP_ALIGN_REPEAT_N"); do
  EPISODES+=",$ROOT/pear/envs/robodojo/configs/pickup_align_all.json"
done
VAL_EPISODES=$ROOT/pear/envs/libero/configs/libero_goal_val.json,$ROOT/pear/envs/robodojo/configs/official_smoke.json
# both families' tools in one tool config, this recipe's own file
TOOL_CONFIG=$ROOT/examples/models/qwen3_8_27b_pear/train/libero_robodojo_tool_config.yaml
# the smallest frame: one 512 x 512 LIBERO frame in the model's tokens
IMAGE_TOKENS=$(model_image_tokens 512 512)
# images in one prompt: only the token budget bounds the observations
LIMIT_IMAGES=$((MAX_MODEL_LEN / IMAGE_TOKENS))
VERL_SRC=$ROOT/3rdparty/verl/src
# the run directory, resolved: every machine's workers write there
RUN_DIR=$(readlink -m "$RUN")
PEAR_LOG_DIR=$RUN_DIR/sim
START_TIME=$(date -u +%Y%m%dT%H%M%SZ)

# ---- parameter arrays ----
DATA_ARGS=(
  # the episode files, turned into rows by pear/agent_loop/dataset.py; a file listed k times weighs k.
  # Validation on libero_goal's trials 40-49 and the first layout of every RoboDojo task
  "data.train_files=[$EPISODES]"
  "data.val_files=[$VAL_EPISODES]"
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESPONSE"
  data.train_batch_size="$BATCH"
)

ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.nnodes="$ROLLOUT_NNODES"
  actor_rollout_ref.rollout.n_gpus_per_node="$GPUS_PER_NODE"
  actor_rollout_ref.rollout.n="$N"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  +actor_rollout_ref.rollout.limit_images="$LIMIT_IMAGES"
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
)

ACTOR_ARGS=(
  # the actor's update: the step's batch as one mini-batch
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH"
)

TRAINER_ARGS=(
  # the run directory: the config takes the workers' paths from it
  run_dir="$RUN_DIR"
  trainer.n_gpus_per_node="$GPUS_PER_NODE"
  trainer.nnodes="$TRAINER_NNODES"
  trainer.validation_data_dir="$RUN/val_dumps"
  trainer.total_training_steps="$STEPS"
  trainer.total_epochs="$EPOCHS"
  trainer.logger="$LOGGER"
  trainer.experiment_name="$(basename "$RUN")"
  trainer.default_local_dir="$RUN/ckpt"
  # Hydra's record of a launch (composed config, overrides) goes to the run, not to the working directory
  hydra.run.dir="$RUN/hydra/$START_TIME"
  # sessions a replica keeps generating, refilled as groups finish; the router pins each to a replica. The replicas'
  # decoding slots, the agent loop's processes and the run's session cap follow from it (run_sizes of base.yaml);
  # within the cap RoboDojo's sessions stay under its servers' Apps: the refill skips a row whose family is at its cap
  run_sizes.sessions_per_replica="$SESSIONS_PER_REPLICA"
)

# what the Ray workers read, next to the variables of pear/configs/base.yaml
ENV_VAR=+ray_kwargs.ray_init.runtime_env.env_vars
# the modules verl imports in every process, the driver's included: they register PEAR's vLLM replica under
# rollout.name=vllm and the model's answer parser under rollout.multi_turn.format
EXTERNAL_MODULES=pear.rollout.server,$MODEL_MODULES
RAY_ARGS=(
  "$ENV_VAR.VERL_USE_EXTERNAL_MODULES='$EXTERNAL_MODULES'"
  # this tree's pear and verl; pear/utils/sitecustomize presizes each process's descriptor table
  "$ENV_VAR.PYTHONPATH='$ROOT/pear/utils/sitecustomize:$ROOT:$VERL_SRC'"
  "$ENV_VAR.VERL_FILE_LOGGER_PATH=$RUN/metrics_$START_TIME.jsonl"
  # LIBERO's servers (pear/envs/libero/configs/tool_config.yaml), one per slot the rollout machines offer as libero
  # (scripts/start_ray_node.sh), each a simulator process in LIBERO's environment, rendering on the CPU
  "$ENV_VAR.LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero/etc/libero"
  "$ENV_VAR.LIBERO_PYTHON=$WORK_DIR/envs/libero/bin/python"
  "$ENV_VAR.LIBERO_MESA_DIR=$WORK_DIR/envs/mesa"
  # RoboDojo's servers (pear/envs/robodojo/configs/tool_config.yaml)
  "$ENV_VAR.ROBODOJO_SERVER_N='$ROBODOJO_SERVER_N'"
  "$ENV_VAR.ROBODOJO_RESPONSE_BUDGET='$ROBODOJO_RESPONSE_BUDGET'"
)
# the tool config reads the Apps of a RoboDojo server from the workers' environment, when given
if [ -n "$ROBODOJO_APPS_PER_GPU" ]; then
  RAY_ARGS+=("$ENV_VAR.ROBODOJO_APPS_PER_GPU='$ROBODOJO_APPS_PER_GPU'")
fi

# ---- launch ----
mkdir -p "$RUN"
# the environment of a GPU machine's processes for the one this recipe starts on the head, which imports vLLM and
# Triton: they compile into the machine's caches, as in a Ray worker
# shellcheck source=../../../../scripts/gpu_process_variables.sh
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
  --target $((BATCH * N)) \
  --group "$N" \
  --run "$RUN" \
  --json \
  >> "$RUN/progress.log" 2>&1 &
PROGRESS_PID=$!
# it may have ended already
trap 'kill "$PROGRESS_PID" 2> /dev/null || true' EXIT
# appended: a resumed run keeps one log
python -m pear.trainer.main \
  "${DATA_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${MODEL_ROLLOUT_ARGS[@]}" \
  "${MODEL_TRAINING_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${ACTOR_ARGS[@]}" \
  "${TRAINER_ARGS[@]}" \
  "${RAY_ARGS[@]}" \
  "$@" 2>&1 | tee -a "$RUN/train.log"
