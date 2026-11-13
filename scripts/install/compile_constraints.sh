#!/usr/bin/env bash
# Writes the constraints.txt of an environment's directory (scripts/install/<environment>, under sim/ for a simulator
# environment): the environment's pins (its constraints.in and, where it has an overrides.txt, the versions that file
# pins) with the SHA-256 of every file of each version, which uv checks at every install. It needs uv and the network,
# and no work directory.
#   bash scripts/install/compile_constraints.sh <environment>
#   environments: verl | libero | libero_plus | libero_pro | robodojo | evalscope | station
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ENVIRONMENT=${1:?verl | libero | libero_plus | libero_pro | robodojo | evalscope | station}
ENVIRONMENT_DIR=$ROOT/scripts/install/$ENVIRONMENT
# a simulator environment's directory
SIM_ENVIRONMENT_DIR=$ROOT/scripts/install/sim/$ENVIRONMENT
# shellcheck source=versions.env
source "$ROOT/scripts/install/versions.env"

# the environment's Python and where its torch comes from, as its target of scripts/install.sh installs it
INDEX_ARGS=()
case "$ENVIRONMENT" in
  verl)
    PYTHON_VERSION=$VERL_PYTHON_VERSION
    INDEX_ARGS=(--torch-backend "$CUDA_TAG" --find-links "$FLASHINFER_CUBIN_WHEELS_URL")
    ;;
  libero)
    ENVIRONMENT_DIR=$SIM_ENVIRONMENT_DIR
    PYTHON_VERSION=$LIBERO_PYTHON_VERSION
    INDEX_ARGS=(--torch-backend cpu)
    ;;
  libero_plus)
    ENVIRONMENT_DIR=$SIM_ENVIRONMENT_DIR
    PYTHON_VERSION=$LIBERO_PLUS_PYTHON_VERSION
    INDEX_ARGS=(--torch-backend cpu)
    ;;
  libero_pro)
    ENVIRONMENT_DIR=$SIM_ENVIRONMENT_DIR
    PYTHON_VERSION=$LIBERO_PRO_PYTHON_VERSION
    INDEX_ARGS=(--torch-backend cpu)
    ;;
  robodojo)
    ENVIRONMENT_DIR=$SIM_ENVIRONMENT_DIR
    PYTHON_VERSION=$ROBODOJO_PYTHON_VERSION
    ;;
  evalscope)
    PYTHON_VERSION=$EVALSCOPE_PYTHON_VERSION
    ;;
  station)
    PYTHON_VERSION=$STATION_PYTHON_VERSION
    INDEX_ARGS=(--torch-backend cpu)
    ;;
  *)
    echo "unknown environment $ENVIRONMENT" >&2
    echo "  environments: verl | libero | libero_plus | libero_pro | robodojo | evalscope | station" >&2
    exit 1
    ;;
esac

# Without the packages' own requirements: the pins are the whole environment. For every platform: the files of both
# machine types. flashinfer-cubin gets no hash: FlashInfer's index, the one place that has it, publishes none.
{
  cat "$ENVIRONMENT_DIR/constraints.in"
  if [ -f "$ENVIRONMENT_DIR/overrides.txt" ]; then
    grep -E '^[A-Za-z0-9_.-]+(\[[^]]*\])?==' "$ENVIRONMENT_DIR/overrides.txt"
  fi
} | UV_NO_CONFIG=1 uv pip compile --quiet --no-deps --generate-hashes --universal --no-annotate \
  --python-version "$PYTHON_VERSION" "${INDEX_ARGS[@]}" \
  --custom-compile-command "bash scripts/install/compile_constraints.sh $ENVIRONMENT" \
  --output-file "$ENVIRONMENT_DIR/constraints.txt" -
