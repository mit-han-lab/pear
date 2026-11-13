#!/usr/bin/env bash
# Installs one target of what PEAR runs with under $WORK_DIR, a work directory outside the login home, at the versions
# of scripts/install/. A work directory the installer makes is owner-only (mode 700); one that exists keeps its mode.
#   WORK_DIR=<work dir> bash scripts/install.sh <target>
# For every environment family (the target's script: scripts/install/<target>/install.sh):
#   verl          the training and rollout environment
#   evalscope     the general evaluation's environment and datasets
#   numactl       numactl and numastat, on a machine that lacks them: libero_run runs it
#   station       the robot station's environment, on the machine a robot is plugged into
#   server        the model server's environment, on the machine of its GPU
# An environment family's own (scripts/install/sim/<target>/install.sh):
#   libero        the LIBERO simulator environment
#   libero_plus   the LIBERO-plus simulator environment
#   libero_pro    the LIBERO-PRO simulator environment
#   robodojo      the Ray environment of RoboDojo's simulator machines
#   robolab       RoboLab's runtime and environment
#   mesa          Mesa, with which the LIBERO families' simulators render: their targets run it
#   imagemagick   ImageMagick, for LIBERO-plus's motion blur: libero_plus runs it
#   libero_run    everything for a LIBERO run, on a GPU machine: the model (scripts/download_model.sh) and the targets
#                 libero, numactl and verl, then a check of what they made; after those on CPU machines it only
#                 finishes and checks
set -euo pipefail

INSTALL_DIR=$(cd "$(dirname "$0")/install" && pwd)
TARGETS=(
  verl evalscope numactl station server libero libero_plus libero_pro robodojo robolab mesa imagemagick libero_run
)

for TARGET in "${TARGETS[@]}"; do
  if [ "$#" -eq 1 ] && [ "$1" = "$TARGET" ]; then
    for TARGET_DIR in "$INSTALL_DIR/$TARGET" "$INSTALL_DIR/sim/$TARGET"; do
      if [ -d "$TARGET_DIR" ]; then
        exec bash "$TARGET_DIR/install.sh"
      fi
    done
  fi
done
echo "usage: WORK_DIR=<work dir> bash scripts/install.sh <target>" >&2
echo "  targets: ${TARGETS[*]}" >&2
exit 1
