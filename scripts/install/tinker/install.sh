#!/usr/bin/env bash
# The Tinker coordinator, $WORK_DIR/envs/tinker: the SDK, Cookbook, CPU tensor operations and Ray, plus PEAR editable.
#   WORK_DIR=<work dir> bash scripts/install.sh tinker
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

TINKER_ENV=$ENVS/tinker
PYTHON=$TINKER_ENV/bin/python
export UV_CONSTRAINT=$INSTALL_DIR/tinker/constraints.txt

require_tools curl getent tar sha256sum
install_uv
create_environment "$TINKER_ENV" "$TINKER_PYTHON_VERSION"
uv pip install --python "$PYTHON" --torch-backend=cpu -r "$ROOT/pyproject.toml" --extra tinker
uv pip install --python "$PYTHON" --no-deps -e "$ROOT"
