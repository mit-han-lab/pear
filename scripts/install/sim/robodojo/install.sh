#!/usr/bin/env bash
# The Ray environment of RoboDojo's simulator machines, $WORK_DIR/envs/robodojo. A machine's Ray runs inside the Isaac
# container, where the training environment's interpreter, built against the host, cannot start: a uv-managed CPython of
# the training environment's version with its Ray and numpy, made at its resolved path, which the container mounts.
#   WORK_DIR=<work dir> bash scripts/install.sh robodojo
#   NVIDIA_GRAPHICS_VERSION=<the machines' driver version>: also extracts the driver's user-space graphics libraries,
#     for machines whose driver is installed with its compute libraries only (x86_64)
#   SIM_RUNTIME_DIR=<directory>: the runtime directory the graphics libraries go into ($WORK_DIR/robodojo_runtime)
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

ROBODOJO_ENV=$(readlink -m "$ENVS")/robodojo
PYTHON=$ROBODOJO_ENV/bin/python
SIM_RUNTIME_DIR=${SIM_RUNTIME_DIR:-$WORK_DIR/robodojo_runtime}
# what Isaac Sim's RTX renderer loads of the driver: the Vulkan ICD's library, the RTX core, the SPIR-V compiler, OptiX
# and NGX, each at the driver's version
GRAPHICS_LIBRARIES=(
  libGLX_nvidia libEGL_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-tls
  libnvidia-rtcore libnvidia-glvkspirv libnvidia-gpucomp libnvoptix libnvidia-ngx
)

# The driver's user-space graphics libraries at exactly the kernel module's version, which Isaac Sim's RTX renderer
# needs: taken out of NVIDIA's runfile, which is run only as the file of the checksum NVIDIA serves with it, into the
# runtime's nvidia-graphics/<version>, which a RoboDojo machine's container mounts for scripts/start_ray_node.sh to
# install.
extract_graphics_libraries() {
  local graphics_dir=$SIM_RUNTIME_DIR/nvidia-graphics/$NVIDIA_GRAPHICS_VERSION
  local runfile=NVIDIA-Linux-x86_64-$NVIDIA_GRAPHICS_VERSION.run
  local file library

  if [ ! -f "$graphics_dir/nvidia_icd.json" ]; then
    RUNFILE_DIR=$(mktemp -d)
    trap 'rm -rf "$RUNFILE_DIR"' EXIT
    for file in "$runfile" "$runfile.sha256sum"; do
      bash "$INSTALL_DIR/download.sh" "$NVIDIA_DRIVERS_URL/$NVIDIA_GRAPHICS_VERSION/$file" "$RUNFILE_DIR/$file"
    done
    (
      cd "$RUNFILE_DIR"
      sha256sum --check --quiet "$runfile.sha256sum"
    )
    sh "$RUNFILE_DIR/$runfile" --extract-only --target "$RUNFILE_DIR/driver" > /dev/null

    # shellcheck disable=SC2174  # the mode is for the runtime directory, not for the parents -p makes
    mkdir -p -m 700 "$SIM_RUNTIME_DIR"
    rm -rf "$graphics_dir.partial"
    mkdir -p "$graphics_dir.partial"
    for library in "${GRAPHICS_LIBRARIES[@]}"; do
      cp "$RUNFILE_DIR/driver/$library.so.$NVIDIA_GRAPHICS_VERSION" "$graphics_dir.partial/"
    done
    cp "$RUNFILE_DIR/driver/libnvidia-api.so.1" "$RUNFILE_DIR/driver/nvoptix.bin" "$graphics_dir.partial/"
    # the runfile's ICD names the library by a placeholder its installer fills in
    sed 's/__NV_VK_ICD__/libGLX_nvidia.so.0/' "$RUNFILE_DIR/driver/nvidia_icd.json" \
      > "$graphics_dir.partial/nvidia_icd.json"
    mv "$graphics_dir.partial" "$graphics_dir"
  fi
}

# the Python builds at their resolved path too, which the container mounts
UV_PYTHON_INSTALL_DIR=$(readlink -m "$UV_PYTHON_INSTALL_DIR")

require_tools curl getent tar sha256sum readlink
install_uv
# a machine joins only a cluster of the same Python and Ray: the training environment's Python, and its Ray and numpy
# at the versions of the environment's own constraints
create_environment "$ROBODOJO_ENV" "$ROBODOJO_PYTHON_VERSION"
uv pip install --python "$PYTHON" --constraint "$INSTALL_DIR/sim/robodojo/constraints.txt" \
  -r "$ROOT/pyproject.toml" --extra robodojo
if [ -n "${NVIDIA_GRAPHICS_VERSION:-}" ]; then
  extract_graphics_libraries
fi
