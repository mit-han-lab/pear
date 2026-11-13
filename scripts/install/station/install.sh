#!/usr/bin/env bash
# The robot station's environment, $WORK_DIR/envs/station, on the machine a robot is plugged into: what pear-station
# (pear/deploy/station) drives the robot, serves its GUI and reaches the models with, the station's tests, and PEAR,
# editable. It needs no GPU. ruckig is compiled here, with the machine's C++ compiler; the YAM's RealSense cameras
# load the machine's libusb-1.0 (Debian and Ubuntu: the package libusb-1.0-0).
#   WORK_DIR=<work dir> bash scripts/install.sh station
set -euo pipefail

# shellcheck source=../common.sh
source "$(dirname "$0")/../common.sh"

STATION_ENV=$ENVS/station
PYTHON=$STATION_ENV/bin/python

# for every install below: the versions the station runs with, each with the hashes of its files, the version that
# replaces what another package requires, and the build tool of what is compiled
export UV_CONSTRAINT=$INSTALL_DIR/station/constraints.txt
export UV_OVERRIDE=$INSTALL_DIR/station/overrides.txt
export UV_BUILD_CONSTRAINT=$INSTALL_DIR/station/build_constraints.txt

require_tools curl getent tar sha256sum git g++
install_uv
create_environment "$STATION_ENV" "$STATION_PYTHON_VERSION"
# in one resolution: the extra station of pyproject.toml and the YAM's SDK at its commit; torch, which lerobot requires
# and the station runs nothing on, as its CPU build
uv pip install --python "$PYTHON" --torch-backend=cpu \
  -r "$ROOT/pyproject.toml" --extra station "i2rt @ git+$I2RT_URL@$I2RT_COMMIT"
# PEAR, editable: the environment's bin/ gets the pear-station command (pyproject.toml)
uv pip install --python "$PYTHON" --no-deps -e "$ROOT"
