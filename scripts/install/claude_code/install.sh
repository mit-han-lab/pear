#!/usr/bin/env bash
# The Claude Code session service's environment, $WORK_DIR/envs/claude_code, on the machine of the model server: the
# Agent SDK with the Claude Code CLI its wheel carries, which the wheel's hashes pin, the service's web server and
# config, its tests, and PEAR, editable. It needs no GPU.
#   WORK_DIR=<work dir> bash scripts/install.sh claude_code
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

CLAUDE_CODE_ENV=$ENVS/claude_code
PYTHON=$CLAUDE_CODE_ENV/bin/python

# for every install below: the versions the service runs with, each with the hashes of its files
export UV_CONSTRAINT=$INSTALL_DIR/claude_code/constraints.txt

require_tools curl getent tar sha256sum
install_uv
create_environment "$CLAUDE_CODE_ENV" "$CLAUDE_CODE_PYTHON_VERSION"
uv pip install --python "$PYTHON" -r "$ROOT/pyproject.toml" --extra claude_code
# PEAR, editable: the service's package, and the station's client whose protocol the service answers
uv pip install --python "$PYTHON" --no-deps -e "$ROOT"
