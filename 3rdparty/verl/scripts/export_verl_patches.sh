#!/usr/bin/env bash
# Rewrites patches/ from the commits of 3rdparty/verl/src since the upstream commit of UPSTREAM, one file per commit.
#   bash 3rdparty/verl/scripts/export_verl_patches.sh
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../UPSTREAM
source "$PROJECT_DIR/UPSTREAM"
PROJECT=verl
UPSTREAM_URL=$VERL_URL
UPSTREAM_COMMIT=$VERL_COMMIT
PATCHES_DIR=$PROJECT_DIR/patches
CHECKOUT_DIR=$PROJECT_DIR/src
# shellcheck source=../../patched_checkout.sh
source "$PROJECT_DIR/../patched_checkout.sh"

export_patches
