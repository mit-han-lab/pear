#!/usr/bin/env bash
# numactl and numastat, with which an LMCache tier is bound to a NUMA node and its placement checked, on a machine
# that lacks them: $WORK_DIR/envs/numactl, from conda-forge, which scripts/gpu_process_variables.sh puts on PATH. The
# target libero_run runs it.
#   WORK_DIR=<work dir> bash scripts/install.sh numactl
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

if command -v numactl > /dev/null && command -v numastat > /dev/null; then
  exit 0
fi

require_tools curl getent sha256sum
install_conda_prefix "$ENVS/numactl" "$INSTALL_DIR/numactl/versions.txt"
"$ENVS/numactl/bin/numactl" --version
