#!/usr/bin/env bash
# The LMCache whose wheel this repository runs with: the upstream commit of UPSTREAM with patches/ applied, one commit
# each, in LMCACHE_SRC, a directory outside the repository. Ends with status 1 unless the checkout is that.
#   LMCACHE_SRC=<directory> bash 3rdparty/LMCache/scripts/checkout_lmcache.sh
#     builds the checkout when it is missing, then compares it with the patches
#   LMCACHE_SRC=<directory> bash 3rdparty/LMCache/scripts/checkout_lmcache.sh --check
#     compares only
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../UPSTREAM
source "$PROJECT_DIR/UPSTREAM"
PROJECT=LMCache
UPSTREAM_URL=$LMCACHE_URL
UPSTREAM_COMMIT=$LMCACHE_COMMIT
PATCHES_DIR=$PROJECT_DIR/patches
CHECKOUT_DIR=${LMCACHE_SRC:?the directory of the LMCache checkout}
# a wheel is built from this checkout: one that differs from the patches is never accepted
CHANGE_IN_PROGRESS=
# shellcheck source=../../patched_checkout.sh
source "$PROJECT_DIR/../patched_checkout.sh"

checkout "${1:-}"
