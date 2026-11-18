#!/usr/bin/env bash
# The robot station (pear/deploy/station) with one model, which a vLLM server on another machine serves: the page's
# service on the machine a robot is plugged into, in this terminal until it is stopped. The station's own file gives
# its robot; this recipe gives the model and the run directory. With CLAUDE_CODE, the page also offers the model
# through the Claude Code session service beside the server (run_claude_code.sh).
#   export WORK_DIR=<work dir> STATION_CONFIG=<the station's file> SERVER=<host:port> NAME=<served name> RUN=<run dir>
#   bash examples/models/qwen3_8_27b_pear/deploy/run_station.sh
set -euo pipefail

# ---- user-adjustable ----
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
STATION_CONFIG=${STATION_CONFIG:?the configuration file of this station}
SERVER=${SERVER:?the model server, host:port}
NAME=${NAME:?the name the model is served under}
RUN=${RUN:?the run directory}
STATION_PYTHON=${STATION_PYTHON:-$WORK_DIR/envs/station/bin/python}
MODEL_LABEL=Qwen3.8-27B # what the page calls the model
# the Claude Code session service (run_claude_code.sh), host:port; empty: the page offers the server alone
CLAUDE_CODE=${CLAUDE_CODE:-}
# ---- end user-adjustable ----

# the model through Claude Code, beside the default one
CLAUDE_CODE_MODEL=()
if [ -n "$CLAUDE_CODE" ]; then
  CLAUDE_CODE_MODEL=(
    models.claude_code.label="$MODEL_LABEL · Claude Code"
    models.claude_code.client_class=pear.deploy.station.agent_loop.clients.claude_code.ClaudeCodeClient
    models.claude_code.client.base_url="http://$CLAUDE_CODE"
    models.claude_code.client.model="$NAME"
  )
fi

# the model the page starts with: the server's, through its OpenAI API
exec "$STATION_PYTHON" -m pear.deploy.station.cli gui --config "$STATION_CONFIG" \
  run_dir="$RUN" \
  default_model=served \
  models.served.label="$MODEL_LABEL" \
  models.served.client_class=pear.deploy.station.agent_loop.clients.vllm.VLLMClient \
  models.served.client.base_url="http://$SERVER/v1" \
  models.served.client.model="$NAME" \
  "${CLAUDE_CODE_MODEL[@]}"
