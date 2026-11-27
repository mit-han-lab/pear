#!/usr/bin/env bash
# Run the Tinker backend with the LIBERO installation under WORK_DIR.
#   WORK_DIR=<work dir> bash examples/tinker/run_libero.sh train --config examples/tinker/stage1.yaml
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
export LIBERO_PYTHON=${LIBERO_PYTHON:-$WORK_DIR/envs/libero/bin/python}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$WORK_DIR/envs/libero/etc/libero}
export LIBERO_MESA_DIR=${LIBERO_MESA_DIR:-$WORK_DIR/envs/mesa}
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
export USER=${USER:-$(id -un)}
cd "$ROOT"
exec "$WORK_DIR/envs/tinker/bin/python" -m pear.backends.tinker "$@"
