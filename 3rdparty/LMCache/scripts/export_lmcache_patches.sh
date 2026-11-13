#!/usr/bin/env bash
# Rewrites patches/ from the commits of the checkout LMCACHE_SRC since the upstream commit of UPSTREAM, one file per
# commit.
#   LMCACHE_SRC=<directory> bash 3rdparty/LMCache/scripts/export_lmcache_patches.sh
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../UPSTREAM
source "$PROJECT_DIR/UPSTREAM"
PROJECT=LMCache
UPSTREAM_URL=$LMCACHE_URL
UPSTREAM_COMMIT=$LMCACHE_COMMIT
PATCHES_DIR=$PROJECT_DIR/patches
CHECKOUT_DIR=${LMCACHE_SRC:?the directory of the LMCache checkout}
# shellcheck source=../../patched_checkout.sh
source "$PROJECT_DIR/../patched_checkout.sh"

export_patches
