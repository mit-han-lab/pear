#!/usr/bin/env bash
# The verl this repository runs with: the upstream commit of UPSTREAM with patches/ applied, one commit each, in
# 3rdparty/verl/src (git-ignored). Ends with status 1 unless the checkout is that.
#   bash 3rdparty/verl/scripts/checkout_verl.sh
#     builds the checkout when it is missing, then compares it with the patches
#   bash 3rdparty/verl/scripts/checkout_verl.sh --check
#     compares only
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../UPSTREAM
source "$PROJECT_DIR/UPSTREAM"
PROJECT=verl
UPSTREAM_URL=$VERL_URL
UPSTREAM_COMMIT=$VERL_COMMIT
PATCHES_DIR=$PROJECT_DIR/patches
CHECKOUT_DIR=$PROJECT_DIR/src
# 1: a checkout that differs from the patches, as while a verl change is made in it, is reported and accepted
CHANGE_IN_PROGRESS=${VERL_CHANGE_IN_PROGRESS:-}
# shellcheck source=../../patched_checkout.sh
source "$PROJECT_DIR/../patched_checkout.sh"

checkout "${1:-}"
