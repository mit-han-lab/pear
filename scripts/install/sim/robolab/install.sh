#!/usr/bin/env bash
# RoboLab's runtime and environment, as its README installs it: the checkout at its pinned commit with its Git LFS
# assets, its uv environment $WORK_DIR/envs/robolab with Isaac Sim and Isaac Lab (the isaac51 extra) from PEAR's lock,
# the libraries Isaac Sim assumes a GPU machine has, and a local copy of the scenes' remote assets. On a CPU machine.
#   WORK_DIR=<work dir> bash scripts/install.sh robolab
#   SIM_RUNTIME_DIR=<directory>: the runtime directory, which scripts/start_ray_node.sh names too
#     ($WORK_DIR/robolab_runtime)
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

ROBOLAB_ENV=$ENVS/robolab
SIM_RUNTIME_DIR=${SIM_RUNTIME_DIR:-$WORK_DIR/robolab_runtime}
CHECKOUT=$SIM_RUNTIME_DIR/RoboLab
# the architecture Ubuntu names the host libraries' packages and their archive by
DEB_ARCH=amd64
if [ "$MACHINE_TYPE" = aarch64 ]; then
  DEB_ARCH=arm64
fi
UBUNTU_ARCHIVE_URL_NAME=UBUNTU_ARCHIVE_URL_$DEB_ARCH

# RoboLab at its pinned commit with its Git LFS assets, and its project file with PEAR's patch: what the lock adds to
# RoboLab's own requirements (scripts/install/sim/robolab/pyproject.patch).
checkout_robolab() {
  if [ ! -d "$CHECKOUT/.git" ]; then
    git clone -q --no-checkout "$ROBOLAB_URL" "$CHECKOUT"
  fi
  git -C "$CHECKOUT" fetch -q origin "$ROBOLAB_COMMIT"
  # forced: the project file as the commit has it, whatever an earlier run's patch left
  git -C "$CHECKOUT" checkout -q --detach --force "$ROBOLAB_COMMIT"
  git -C "$CHECKOUT" lfs install --local > /dev/null
  git -C "$CHECKOUT" lfs pull
  patch --fuzz=0 --no-backup-if-mismatch --reject-file=- -p1 -d "$CHECKOUT" < "$INSTALL_DIR/sim/robolab/pyproject.patch"
}

# What Isaac Sim assumes a GPU machine has, and one with the driver's own libraries only lacks (libGLU and libOpenGL for
# its MDL SDK, libGL for Kit's renderer, libEGL for NVIDIA's Vulkan driver): Ubuntu's packages, whose content the
# checksums of host_debs.sha256 fix, not the archive they come from; their libraries go to the runtime's host/lib.
install_host_libraries() {
  local debs=$SIM_RUNTIME_DIR/downloads/debs
  local sha256 deb

  mkdir -p "$SIM_RUNTIME_DIR/host/lib" "$debs"
  UNPACK_DIR=$(mktemp -d)
  trap 'rm -rf "$UNPACK_DIR"' EXIT
  grep "_$DEB_ARCH.deb\$" "$INSTALL_DIR/sim/robolab/host_debs.sha256" > "$debs/SHA256SUMS"
  while read -r sha256 deb; do
    if [ ! -f "$debs/$deb" ]; then
      download_checked "${!UBUNTU_ARCHIVE_URL_NAME}/$deb" "$sha256" "$debs/$deb"
    fi
  done < "$debs/SHA256SUMS"
  (
    cd "$debs"
    sha256sum --check --quiet SHA256SUMS
  )

  while read -r _ deb; do
    dpkg-deb -x "$debs/$deb" "$UNPACK_DIR"
  done < "$debs/SHA256SUMS"
  cp -a "$UNPACK_DIR/usr/lib/$MACHINE_TYPE-linux-gnu/"*.so.* "$SIM_RUNTIME_DIR/host/lib/"
}

require_tools curl getent tar sha256sum git git-lfs patch dpkg-deb
install_uv
# shellcheck disable=SC2174  # the mode is for the runtime directory, not for the parents -p makes
mkdir -p -m 700 "$SIM_RUNTIME_DIR"
echo "== 1. RoboLab's checkout at $ROBOLAB_COMMIT with its LFS assets"
checkout_robolab
echo "== 2. the environment (uv sync --extra isaac51, scripts/install/sim/robolab/uv.lock)"
# RoboLab commits no lock, so a sync would resolve anew, nightly builds included: PEAR's lock pins every package, and
# the sync installs exactly it. With RoboLab's own uv settings (its indexes, its build requirements), not without any.
cp "$INSTALL_DIR/sim/robolab/uv.lock" "$CHECKOUT/uv.lock"
(
  cd "$CHECKOUT"
  env -u UV_NO_CONFIG UV_PROJECT_ENVIRONMENT="$ROBOLAB_ENV" \
    uv sync --frozen --extra isaac51 --python "$ROBOLAB_PYTHON_VERSION"
)
echo "== 3. the host libraries (scripts/install/sim/robolab/host_debs.sha256)"
install_host_libraries
echo "== 4. the scenes' remote assets (scripts/install/sim/robolab/remote_assets.txt)"
# on NVIDIA's content buckets: the Apps read them from a local copy
"$ROBOLAB_ENV/bin/python" "$INSTALL_DIR/sim/robolab/download_remote_assets.py" \
  --urls "$INSTALL_DIR/sim/robolab/remote_assets.txt" \
  --checksums "$INSTALL_DIR/sim/robolab/remote_assets.sha256" \
  --destination "$SIM_RUNTIME_DIR/omni_content" \
  --config "$SIM_RUNTIME_DIR/omni_config/omniverse.toml"
