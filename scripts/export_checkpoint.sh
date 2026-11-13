#!/usr/bin/env bash
# Writes a checkpoint of a run as a Hugging Face model directory, which a recipe takes as MODEL: verl's model merger,
# with the weights the run's replicas keep in fp32 kept in fp32, as the config Hydra recorded at the run's latest
# launch names them (rollout.fp32_modules, which pear/utils/model_merger.py reads). On a machine with the memory for the
# model:
#   WORK_DIR=<work dir> bash scripts/export_checkpoint.sh <run dir>/ckpt/global_step_<step> <model directory>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECKPOINT=${1:?the checkpoint, <run dir>/ckpt/global_step_<step>}
MODEL_DIR=${2:?the model directory to write}

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
VERL_ENV=${VERL_ENV:-$WORK_DIR/envs/verl}
# the config of the checkpoint's run, for a checkpoint that is no longer in its run directory
RUN_CONFIG=${RUN_CONFIG:-}
# ---- end user-adjustable ----

if [ -z "$RUN_CONFIG" ]; then
  # a recipe's run directory holds ckpt/global_step_<step> and, of every launch, hydra/<start time>/.hydra: the start
  # times sort by date, so the last config is the latest launch's
  LAUNCH_CONFIGS=("$CHECKPOINT"/../../hydra/*/.hydra/config.yaml)
  RUN_CONFIG=${LAUNCH_CONFIGS[${#LAUNCH_CONFIGS[@]} - 1]}
fi
if [ ! -f "$RUN_CONFIG" ]; then
  echo "no config of the checkpoint's run at $RUN_CONFIG: name the run's config as RUN_CONFIG"
  exit 1
fi
echo "the fp32 modules of $RUN_CONFIG"

# this tree's pear and verl: verl's merge of the checkpoint's actor
PYTHONPATH=$ROOT:$ROOT/3rdparty/verl/src exec "$VERL_ENV/bin/python" -m pear.utils.model_merger --config "$RUN_CONFIG" \
  merge --backend fsdp --trust-remote-code --local_dir "$CHECKPOINT/actor" --target_dir "$MODEL_DIR"
