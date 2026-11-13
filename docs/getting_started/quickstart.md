# Quick Start

**What you need**

- PEAR installed: [Installation](installation.md)
- A cluster of 32 GPU nodes that see the same `WORK_DIR`:

  | GPUs per node | `GPU` | Trainer nodes | Rollout nodes |
  | --- | --- | --- | --- |
  | 8 H100 | `h100` | 8 | 24 |
  | 4 GB300 | `gb300` | 8 | 24 |

**What you will do**

- Start a Ray cluster
- Launch stage 1 of the training recipe and watch it
- Evaluate a checkpoint on LIBERO

## Start Ray

Run the script once on every node, the head first. It stays in the foreground until Ray stops:

```bash
export WORK_DIR=<work dir>
bash scripts/start_ray_node.sh trainer                   # the first node: a trainer node and the Ray head
bash scripts/start_ray_node.sh trainer <head ip>:6379    # the other trainer nodes
bash scripts/start_ray_node.sh rollout <head ip>:6379    # the rollout nodes: the vLLM engines and the simulators
```

## Train

On the head, launch stage 1: RL on LIBERO-Goal, -Spatial and -Object from the base model, 128 groups of 8 rollouts
per step.

```bash
export WORK_DIR=<work dir> RAY_ADDRESS=<head ip>:6379
export GPU=h100    # H100 and H200. On GB300: GPU=gb300

TRAINER_NNODES=8 ROLLOUT_NNODES=24 RUN=$WORK_DIR/runs/stage1 bash examples/models/qwen3_8_27b_pear/train/run_stage1.sh
```

| Variable | Meaning |
| --- | --- |
| `GPU` | `h100` for H100 and H200, `gb300` for GB300 |
| `TRAINER_NNODES` | The nodes started as `trainer` |
| `ROLLOUT_NNODES` | The nodes started as `rollout` |
| `RUN` | The run directory |

The run goes on until you stop it. The same command on an existing `RUN` resumes it from its last checkpoint.

## Watch a Run

A run writes into `RUN`:

| Path | Content |
| --- | --- |
| `train.log` | The trainer's log |
| `progress.log` | One line every 10 seconds: the step, the groups ready, the successes, the engines' throughput |
| `ckpt/` | The checkpoints |
| `sim/` | The simulators' and the episodes' logs |

The same progress, live in a terminal:

```bash
RAY_ADDRESS=<head ip>:6379 $WORK_DIR/envs/verl/bin/python -m pear.monitor.progress_panel \
  --target 1024 --group 8 --run $WORK_DIR/runs/stage1
```

`--target` is the rollouts a step trains on and `--group` the rollouts per group.

## Evaluate

Export a checkpoint to a model directory:

```bash
bash scripts/export_checkpoint.sh $WORK_DIR/runs/stage1/ckpt/global_step_<step> <model dir>
```

An evaluation runs on a cluster whose GPU nodes all started as `rollout`:

```bash
bash scripts/start_ray_node.sh rollout                   # the first node: the Ray head
bash scripts/start_ray_node.sh rollout <head ip>:6379    # the other nodes
```

On the head, evaluate the model on LIBERO's 2,000 episodes:

```bash
export WORK_DIR=<work dir> RAY_ADDRESS=<head ip>:6379
export GPU=h100    # H100 and H200. On GB300: GPU=gb300

ROLLOUT_NNODES=32 MODEL=<model dir> EVAL_STEP=<step> RUN=$WORK_DIR/runs/eval bash examples/eval/eval_libero.sh
```

The records and the summary land in `$RUN/val_dumps/eval_<step>.jsonl` and `$RUN/val_dumps/eval_<step>_summary.json`.

## Next Steps

- [Robot-Use RL with Qwen3.8-27B](../../examples/models/qwen3_8_27b_pear/train/README.md): stages 2 and 3
- [Evaluation](../../examples/eval/README.md): the other benchmarks
