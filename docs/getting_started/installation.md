# Installation

PEAR installs with one script, `scripts/install.sh`. It builds everything under one work directory, and every version
it installs is pinned in the repository.

## Requirements

- **GPU**: NVIDIA GB300, H100 or H200, with a driver for CUDA 13. The model server also runs on an RTX 5090; the
  robot station needs no GPU
- **OS**: Linux on x86_64 or aarch64
- **Tools**: bash 4.4 or newer, `git`, `curl`, `tar`, `patch`, `perl`, `gcc` and `g++`
- **Network** while installing: PyPI, PyTorch's index, GitHub, conda-forge, the Hugging Face Hub and FlashInfer's
  wheel index
- **A shared work directory**: every node of the cluster sees the same `WORK_DIR`

The installer brings its own uv, micromamba and Python.

## Install

On a GPU node:

```bash
git clone https://github.com/mit-han-lab/pear.git && cd pear
export WORK_DIR=<a directory every node sees>
bash scripts/install.sh libero_run
```

`libero_run` installs everything a LIBERO run needs, then checks it:

1. The base model, from the Hugging Face Hub
2. The LIBERO simulator environment
3. The training and rollout environment: PyTorch, vLLM, the patched veRL and PEAR
4. A check on the GPU: the environment's tests, then one action in a simulator

A successful install ends with the simulator's reply:

```text
LiberoEnv: <n> native actions in <t> s, frames [...]
```

### Build on CPU nodes

Only the check needs a GPU. Everything else can be built on CPU nodes first:

```bash
bash scripts/download_model.sh
bash scripts/install.sh libero
bash scripts/install.sh verl         # with CUDA_ARCH=9.0 when the cluster's GPUs are H100 or H200
bash scripts/install.sh libero_run   # on a GPU node: finds the rest built, and checks it
```

## Targets

| Target | Installs |
| --- | --- |
| `libero_run` | Everything a LIBERO run needs, then a check |
| `verl` | The training and rollout environment |
| `libero`, `libero_plus`, `libero_pro` | The simulator environment of LIBERO, LIBERO-Plus or LIBERO-Pro |
| `robodojo` | The Ray environment of RoboDojo's simulator nodes |
| `robolab` | RoboLab's runtime and environment |
| `evalscope` | The environment and the datasets of the general benchmarks |
| `numactl` | numactl, on a node that lacks it |
| `station` | The robot station's environment, on the computer of its robot ([Deploying Qwen3.8-27B-PEAR](../../examples/models/qwen3_8_27b_pear/deploy/README.md#drive-a-robot)) |
| `server` | The model server's environment, on the machine of its GPU ([Deploying Qwen3.8-27B-PEAR](../../examples/models/qwen3_8_27b_pear/deploy/README.md)) |

RoboDojo and RoboLab run Isaac Sim and need more setup. Their guides are coming soon.

## Options

| Variable | Target | Meaning |
| --- | --- | --- |
| `WORK_DIR` | all | The work directory. Required |
| `MODEL_FILE` | `libero_run`, `scripts/download_model.sh` | The model to download. Default: `examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh` |
| `CUDA_ARCH` | `verl`, `libero_run` | The GPU to compile for when it is not the node's own: `10.3` for GB300, `9.0` for H100 and H200. A node without a GPU compiles for `10.3` |
| `MAX_JOBS` | `verl`, `libero_run` | Parallel compile jobs. Default: one per CPU, up to 64. Each needs about 5 GB of memory |
| `NO_GPU=1` | `verl` | Skip what is compiled for a GPU. Enough for the CPU tests |

## Update

A target that runs again rebuilds only what changed. After an update of the repository that touches `pyproject.toml`,
`scripts/install/` or `3rdparty/`, run:

```bash
bash scripts/install.sh verl
```

Do not run a target on an environment that a run is using.

## Next Steps

- [Quick Start](quickstart.md)
