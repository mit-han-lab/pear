#!/usr/bin/env bash
# Mesa (llvmpipe and the EGL software device), with which the LIBERO families' simulator processes render on the CPU,
# $WORK_DIR/envs/mesa, from conda-forge: machines ship no OSMesa and their own Mesa lacks the EGL device platform MuJoCo
# needs. With it come the GL entry-point libraries PyOpenGL loads (libGL, libOpenGL), which a machine without libglvnd
# lacks. The targets libero, libero_plus and libero_pro run it.
#   WORK_DIR=<work dir> bash scripts/install.sh mesa
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

require_tools curl getent sha256sum
install_conda_prefix "$ENVS/mesa" "$INSTALL_DIR/sim/mesa/versions.txt" "$INSTALL_DIR/sim/conda_common_versions.txt"
