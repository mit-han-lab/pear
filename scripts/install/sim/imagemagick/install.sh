#!/usr/bin/env bash
# ImageMagick for Wand, with which LIBERO-plus blurs a moving camera's frames, $WORK_DIR/envs/imagemagick, from
# conda-forge. Its libraries need a newer libstdc++ than some machines have: the simulator processes get conda-forge's
# from the Mesa prefix (LiberoPlusEnv.setup_env). The target libero_plus runs it.
#   WORK_DIR=<work dir> bash scripts/install.sh imagemagick
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

require_tools curl getent sha256sum
install_conda_prefix "$ENVS/imagemagick" "$INSTALL_DIR/sim/imagemagick/versions.txt" \
  "$INSTALL_DIR/sim/conda_common_versions.txt"
