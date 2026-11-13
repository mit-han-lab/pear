#!/usr/bin/env bash
# Writes scripts/install/sim/robolab/uv.lock, the lock of RoboLab's environment: RoboLab's project file at its pinned
# commit with PEAR's patch (scripts/install/sim/robolab/pyproject.patch), locked by uv next to the present lock, whose
# versions stay. It needs uv, git and the network, and no work directory.
#   bash scripts/install/sim/robolab/lock.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
LOCK=$ROOT/scripts/install/sim/robolab/uv.lock
# shellcheck source=../../versions.env
source "$ROOT/scripts/install/versions.env"

PROJECT_DIR=$(mktemp -d)
trap 'rm -rf "$PROJECT_DIR"' EXIT
# the project file alone: a lock reads nothing else of the checkout, whose assets are large
git clone -q --no-checkout --filter=blob:none "$ROBOLAB_URL" "$PROJECT_DIR"
git -C "$PROJECT_DIR" checkout -q "$ROBOLAB_COMMIT" -- pyproject.toml
patch --fuzz=0 -p1 -d "$PROJECT_DIR" < "$ROOT/scripts/install/sim/robolab/pyproject.patch"
cp "$LOCK" "$PROJECT_DIR/uv.lock"
(
  cd "$PROJECT_DIR"
  uv lock
)
cp "$PROJECT_DIR/uv.lock" "$LOCK"
