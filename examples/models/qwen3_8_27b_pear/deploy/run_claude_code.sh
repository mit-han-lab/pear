#!/usr/bin/env bash
# The Claude Code session service of the model server on this machine (pear/deploy/harness/claude_code/main.py), in this
# terminal until it is stopped: beside run_server.sh, in the environment of scripts/install.sh claude_code. The robot
# station's Claude Code model reaches it, and its Claude Code sessions reach the model through it. Extra arguments are
# Hydra overrides:
#   export WORK_DIR=<work dir> RUN=<run dir>
#   bash examples/models/qwen3_8_27b_pear/deploy/run_claude_code.sh
set -euo pipefail

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUN=${RUN:?the run directory}
# the model server, host:port, as run_server.sh serves it
SERVER=${SERVER:-127.0.0.1:8000}
NAME=${NAME:-qwen3_8_27b}
PORT=${PORT:-8001}
CLAUDE_CODE_ENV=${CLAUDE_CODE_ENV:-$WORK_DIR/envs/claude_code}
# ---- end user-adjustable ----

START_TIME=$(date -u +%Y%m%dT%H%M%SZ)

exec "$CLAUDE_CODE_ENV/bin/python" -m pear.deploy.harness.claude_code.main "port=$PORT" "upstream=http://$SERVER" \
  "model=$NAME" "run_dir=$RUN/claude_code" \
  hydra.run.dir="$RUN/claude_code/hydra/$START_TIME" "$@"
