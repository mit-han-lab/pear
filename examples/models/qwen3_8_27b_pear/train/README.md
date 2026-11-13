# Robot-Use RL with Qwen3.8-27B

Three stages of RL in simulation teach [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to use robots. Each
stage is one recipe:

| Stage | Recipe | Trains on | Starts from |
| --- | --- | --- | --- |
| 1 | `run_stage1.sh` | LIBERO-Goal, -Spatial and -Object | The base model |
| 2 | `run_stage2.sh` | All four LIBERO suites | A checkpoint of stage 1 |
| 3 | `run_stage3.sh` | LIBERO and RoboDojo's general_pickup and align_blocks | A checkpoint of stage 2 |

## Hardware

Our runs use 32 nodes:

| GPUs per node | `GPU` | Stages 1 and 2 | Stage 3 |
| --- | --- | --- | --- |
| 8 H100 | `h100` | 8 trainer, 24 rollout | 8 trainer, 16 rollout, 8 RoboDojo |
| 4 GB300 | `gb300` | 8 trainer, 24 rollout | 8 trainer, 18 rollout, 6 RoboDojo |

## Train

Run a recipe on the head of a Ray cluster: [Quick Start](../../../../docs/getting_started/quickstart.md) shows how to
start one. Each stage starts from a model directory exported from a checkpoint of the stage before:

```bash
export WORK_DIR=<work dir> RAY_ADDRESS=<head ip>:6379
export GPU=h100    # H100 and H200. On GB300: GPU=gb300

TRAINER_NNODES=8 ROLLOUT_NNODES=24 RUN=<run 1> bash examples/models/qwen3_8_27b_pear/train/run_stage1.sh
bash scripts/export_checkpoint.sh <run 1>/ckpt/global_step_<step> <model 1>

TRAINER_NNODES=8 ROLLOUT_NNODES=24 MODEL=<model 1> RUN=<run 2> bash examples/models/qwen3_8_27b_pear/train/run_stage2.sh
bash scripts/export_checkpoint.sh <run 2>/ckpt/global_step_<step> <model 2>

# stage 3 on H100: the table below has its sizes on GB300
TRAINER_NNODES=8 ROLLOUT_NNODES=16 SIM_NNODES=8 \
  MODEL=<model 2> RUN=<run 3> bash examples/models/qwen3_8_27b_pear/train/run_stage3.sh
```

Stage 3 on each GPU:

| GPUs per node | `TRAINER_NNODES` | `ROLLOUT_NNODES` | `SIM_NNODES` | `ROBODOJO_APPS_PER_GPU` |
| --- | --- | --- | --- | --- |
| 8 H100 | 8 | 16 | 8 | 8, the default |
| 4 GB300 | 8 | 18 | 6 | 24 |

The Apps cap RoboDojo's sessions in flight. These sizes give RoboDojo half of a run's sessions: `SIM_NNODES` x GPUs
per node x `ROBODOJO_APPS_PER_GPU` Apps for `ROLLOUT_NNODES` x 2 x `SESSIONS_PER_REPLICA` sessions.

## Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | Required | `h100` for H100 and H200, `gb300` for GB300 |
| `TRAINER_NNODES` | Required | The nodes started as `trainer` |
| `ROLLOUT_NNODES` | Required | The nodes started as `rollout` |
| `SIM_NNODES` | Required in stage 3 | The RoboDojo simulator nodes |
| `ROBODOJO_APPS_PER_GPU` | 8 | The Apps on a simulator GPU, an episode each |
| `SESSIONS_PER_REPLICA` | 64 in stage 1, 32 after | The sessions a vLLM replica keeps generating |
| `MODEL` | The base model in stage 1, required after | The model directory to start from |
| `RUN` | Required | The run directory. An existing one resumes from its last checkpoint |
| `STEPS` | 1000 | The most steps a run takes. A run goes on until you stop it |
| `BATCH`, `N` | 128, 8 | The groups per step and the rollouts per group |
| `MAX_RESPONSE` | 160000 in stage 1, 256000 after | The response tokens of an episode |

Extra arguments are Hydra overrides.

## Evaluate

[Evaluation](../../../eval/README.md) has one recipe per benchmark, for a model directory exported from a checkpoint.
