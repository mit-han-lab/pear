# Evaluation

One recipe per benchmark:

| Recipe | Benchmark | Episodes |
| --- | --- | --- |
| `eval_libero.sh` | LIBERO | 2,000 |
| `eval_libero_plus.sh` | LIBERO-Plus | 10,030 |
| `eval_libero_pro.sh` | LIBERO-Pro | 10,000 |
| `eval_robodojo.sh` | RoboDojo's general_pickup and align_blocks | 328 |
| `eval_robolab.sh` | RoboLab-120 | 120 |
| `eval_general.sh` | Nine reasoning, coding and vision benchmarks | |

## Robotic Benchmarks

Run a recipe on the head of a Ray cluster whose GPU nodes all started as `rollout`:
[Quick Start](../../docs/getting_started/quickstart.md) shows how to start one.

```bash
export WORK_DIR=<work dir> RAY_ADDRESS=<head ip>:6379
export GPU=h100    # H100 and H200. On GB300: GPU=gb300

ROLLOUT_NNODES=32 MODEL=<model dir> EVAL_STEP=<step> RUN=<run dir> bash examples/eval/eval_libero.sh

# RoboDojo on H100. On GB300: ROBODOJO_APPS_PER_GPU=24 ROLLOUT_NNODES=26 SIM_NNODES=6
ROLLOUT_NNODES=24 SIM_NNODES=8 \
  MODEL=<model dir> EVAL_STEP=<step> RUN=<run dir> bash examples/eval/eval_robodojo.sh
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPU` | Required | `h100` for H100 and H200, `gb300` for GB300 |
| `ROLLOUT_NNODES` | Required | The GPU nodes of the cluster, all started as `rollout` |
| `SIM_NNODES` | Required for RoboDojo and RoboLab | The simulator nodes |
| `ROBODOJO_APPS_PER_GPU`, `ROBOLAB_APPS_PER_GPU` | 8, 10 | The Apps on a simulator GPU, an episode each |
| `MODEL` | Required | The model directory to evaluate |
| `MODEL_FILE` | `examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh` | The model's settings on each GPU |
| `EVAL_STEP` | Required | An integer that names the results, such as the checkpoint's step |
| `EPISODES` | The benchmark's full set | The episodes to run |
| `N` | 1 | The rollouts per episode |

The Apps cap the sessions in flight: `SIM_NNODES` x GPUs per node x Apps per GPU. Extra arguments are Hydra overrides.

The records and the summary land in `$RUN/val_dumps/eval_<step>.jsonl` and `$RUN/val_dumps/eval_<step>_summary.json`.

## General Benchmarks

The general benchmarks run through [EvalScope](https://github.com/modelscope/evalscope), on vLLM servers of their own.
Install them with `bash scripts/install.sh evalscope`, then:

```bash
export WORK_DIR=<work dir> MODEL=<model dir> NAME=<served name> RUN=<run dir>
export GPU=h100    # H100 and H200. On GB300: GPU=gb300

bash examples/eval/eval_general.sh serve                             # on every GPU node, left running
bash examples/eval/eval_general.sh run <host>:8000,<host>:8001,...   # on one of them
```
