#!/bin/bash
# /isaac-sim/python.sh with `exec`: the same variables, but Python replaces this shell. A server signals and
# reaps its Apps directly; python.sh would keep bash in between and turn every nonzero exit into 1.
set -e
ISAAC=${ISAAC_SIM_ROOT:-/isaac-sim}
export CARB_APP_PATH=$ISAAC/kit ISAAC_PATH=$ISAAC EXP_PATH=$ISAAC/apps
# shellcheck source=/dev/null
source "$ISAAC/setup_python_env.sh"
# python.sh's preload of Kit's libcarb, ahead of the caller's (pear/envs/robodojo/apps.py preloads the Apps' inotify)
export RESOURCE_NAME=IsaacSim LD_PRELOAD="$ISAAC/kit/libcarb.so${LD_PRELOAD:+ $LD_PRELOAD}"
exec "${PYTHONEXE:-$ISAAC/kit/python/bin/python3}" "$@"
