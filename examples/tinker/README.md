# Tinker

Run PEAR's sampling and LoRA training on Tinker, with simulators on your own machines. The backend uses the
Tinker Cookbook's environment and sampler interfaces with PEAR's prompts, action parser and rewards.
The coordinator and LIBERO run on CPUs; RoboDojo and RoboLab require GPUs for their simulators.
Camera frames use lossless PNG transport. `model_revision` pins tokenizer and image processor assets to PEAR's
Hugging Face revision; Tinker does not expose a base-weight revision through its model-info API.

## Install and run

From the repository root on Linux:

```bash
export WORK_DIR=<installation directory>
bash scripts/install.sh tinker
"$WORK_DIR/envs/tinker/bin/tinker" auth login
bash scripts/install.sh libero

bash examples/tinker/run_libero.sh preflight --config examples/tinker/libero.yaml
bash examples/tinker/run_libero.sh train --config examples/tinker/libero.yaml
bash examples/tinker/run_libero.sh eval --config examples/tinker/libero.yaml
```

The [small LIBERO recipe](libero.yaml) collects eight trajectories through two CPU workers and stops after one
update. It trains on two trials of "turn on the stove" and evaluates on a held-out trial. Groups with constant
rewards are replaced; this recipe stops if neither of its two groups can supply an update.
`preflight` checks simulator resources and runs a reset, action and camera capture without calling Tinker.
Use `check` for configuration checks without starting simulators.

The runner uses the LIBERO installation under WORK_DIR. Existing installations can set LIBERO_PYTHON,
LIBERO_CONFIG_PATH and LIBERO_MESA_DIR. Authentication uses the saved login or TINKER_API_KEY.
Checkpoints, episodes, metrics and token usage are written under `run_dir`.

Rerunning `train` resumes the latest checkpoint and optimizer. To continue the one-update example:

```bash
bash examples/tinker/run_libero.sh train --config examples/tinker/libero.yaml steps=2
```

Evaluation uses the latest checkpoint in the same run directory, or the base model if none exists.
Pass `--checkpoint` to select another Tinker checkpoint. Native training resamples pending trajectories after a
restart because Tinker's sampling records belong to the previous service session.
Native checkpoints keep task order, group metadata and discard counts; other objectives retain trajectories for
replay. Set `dump_training_batch=true` to save the latest full batch in `checkpoints/batch.pkl` for debugging.
That optional dump is separate from the restart checkpoint.

## Training recipes

| Recipe | Tasks |
|---|---|
| [Stage 1](stage1.yaml) | LIBERO goal, spatial and object tasks |
| [Stage 2](stage2.yaml) | All LIBERO tasks |
| [Stage 3](stage3.yaml) | LIBERO and RoboDojo |

Pass one of these files to the same runner. Stage 2 and stage 3 require `init_checkpoint=<Tinker path>` from the
preceding stage; they load its weights into a fresh optimizer and use separate run directories. The learning rate
is a LoRA starting point to tune. See the [simulator setup](../models/qwen3_8_27b_pear/train/README.md) for RoboDojo.

The default objective is native StableReinforce with sampler–learner KL disabled. PEAR's group-standardized
advantages, observation masking, full-batch token normalization and constant-reward filtering are preserved.
LoRA with StableReinforce has not been shown to reproduce the original full-model PEAR training results.

`batch_groups` controls groups per update, `max_sessions` limits concurrent episodes, and `prefetch_groups=0`
collects only the next batch. A group can be collected through fewer sessions than its `group_size`.
`max_staleness` limits how many updates a group may lag behind the current policy.

For PEAR's explicit top-32 score correction, set `objective=pear 'loss_config={}' allow_head_continuations=true`.
This requires short sampling continuations and repeated training passes, which substantially increase cost.
Other Tinker losses are `objective=importance_sampling 'loss_config={}'` and
`objective=ppo 'loss_config={}'`. Changing the objective requires a new run directory; use `init_checkpoint`
to carry over weights.

## Simulator cluster

For an existing Ray cluster, add `ray_address="$HEAD_ADDRESS"` to the training command. Otherwise start a CPU
head and attach workers, running each command on its respective machine:

```bash
NUM_CPUS=2 OBJECT_STORE_GB=2 bash scripts/start_ray_node.sh cpu
NUM_CPUS=8 LIBERO_SERVERS_PER_NODE=4 bash scripts/start_ray_node.sh libero "$HEAD_ADDRESS"

# On a machine with RoboDojo installed; use the robolab role for RoboLab.
RAY_ENV="$WORK_DIR/envs/tinker" bash scripts/start_ray_node.sh robodojo "$HEAD_ADDRESS"
```

CPU roles require NUM_CPUS to be set to the slots allocated to that node. Nodes need the same PEAR checkout and
compatible Ray/Python versions. Set WORK_DIR on each machine. Isaac roles keep their original Ray environment;
set RAY_ENV explicitly to use the Tinker environment, as above.
For stage 3, set ROBODOJO_SERVER_N and ROBODOJO_APPS_PER_GPU to the allocated simulator capacity.
Isaac workers can also serve LIBERO with `SIM_LIBERO_CPUS=2 LIBERO_SERVERS_PER_NODE=2` when LIBERO is installed;
leave enough CPU capacity for Isaac. A run owns the named simulator pools on its Ray cluster.

## Docker

The [CPU image](Dockerfile) installs the coordinator and LIBERO:

```bash
docker build -f examples/tinker/Dockerfile -t pear-tinker .
docker run --rm -it --shm-size=3g \
  -v "$HOME/.tinker:/root/.tinker:ro" -v pear-tinker-runs:/runs \
  pear-tinker train --config examples/tinker/libero.yaml run_dir=/runs/libero
```

The credential mount uses the host's saved login. The volume keeps checkpoints and logs across container runs;
replace `train` with `eval` to evaluate the saved policy.
