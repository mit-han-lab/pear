#!/usr/bin/env bash
# The model server's environment, $WORK_DIR/envs/server: what pear/deploy/server serves a model with (vLLM, with the
# torch and FlashInfer it brings, Hydra and uvloop), FlashInfer's precompiled kernels and the CUDA toolkit with which
# the model families' kernels compile at the server's first start, with the machine's C++ compiler; every package at
# the training environment's version, and PEAR, editable.
#   WORK_DIR=<work dir> bash scripts/install.sh server
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

SERVER_ENV=$ENVS/server
PYTHON=$SERVER_ENV/bin/python

require_tools curl getent tar sha256sum
install_uv
create_environment "$SERVER_ENV" "$SERVER_PYTHON_VERSION"
# in one resolution: the extra server of pyproject.toml at the training environment's pins, with their hashes, and at
# its overrides; torch's CUDA 13 build; FlashInfer's precompiled kernels from FlashInfer's own index
uv pip install --python "$PYTHON" --constraint "$INSTALL_DIR/verl/constraints.txt" \
  --override "$INSTALL_DIR/verl/overrides.txt" --torch-backend="$CUDA_TAG" --find-links "$FLASHINFER_CUBIN_WHEELS_URL" \
  -r "$ROOT/pyproject.toml" --extra server
# where the environment's CUDA wheels put the toolkit, among the packages of its Python
CUDA_TOOLKIT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
link_cuda_toolkit "$CUDA_TOOLKIT"
# PEAR, editable, as in every environment that runs its code
uv pip install --python "$PYTHON" --no-deps -e "$ROOT"
