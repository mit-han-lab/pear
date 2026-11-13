#!/usr/bin/env bash
# The LIBERO-plus simulator environment for the family's server processes, $WORK_DIR/envs/libero_plus: LIBERO-plus at
# its pinned commit with its scene assets, robosuite and MuJoCo, a CPU-only torch, Mesa (the target mesa), with which
# the processes render on the CPU, and ImageMagick (the target imagemagick) for its motion blur. The three LIBERO
# families are all the package `libero`, so each has its own environment and its own checkout.
#   WORK_DIR=<work dir> bash scripts/install.sh libero_plus
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

LIBERO_PLUS_ENV=$ENVS/libero_plus
PYTHON=$LIBERO_PLUS_ENV/bin/python
CONSTRAINTS=$INSTALL_DIR/sim/libero_plus/constraints.txt
ROBOSUITE_PATCH=$INSTALL_DIR/sim/libero_plus/robosuite_file_log.patch
CHECKOUT=$WORK_DIR/src/$(basename "$LIBERO_PLUS_URL" .git)
# the libero package inside the checkout, which holds its tasks, initial states and assets
LIBERO_PACKAGE=$CHECKOUT/libero/libero
# the episode of the check
CHECK_ENV_ARGS='{"suite": "libero_spatial", "task_id": 0, "trial_id": 0}'

# The environment's packages (the extra libero_plus of pyproject.toml) at the versions of its constraints, in one
# resolution, so that numpy<2 (LIBERO, robosuite) holds throughout; then LIBERO-plus at its pinned commit, editable.
install_packages() {
  create_environment "$LIBERO_PLUS_ENV" "$LIBERO_PLUS_PYTHON_VERSION"
  if [ ! -d "$CHECKOUT/.git" ]; then
    git clone -q "$LIBERO_PLUS_URL" "$CHECKOUT"
  fi
  git -C "$CHECKOUT" fetch -q origin "$LIBERO_PLUS_COMMIT"
  git -C "$CHECKOUT" checkout -q "$LIBERO_PLUS_COMMIT"
  echo "$(basename "$CHECKOUT"): $(git -C "$CHECKOUT" log --oneline -1)"

  uv pip install --python "$PYTHON" --torch-backend=cpu --constraint "$CONSTRAINTS" \
    -r "$ROOT/pyproject.toml" --extra libero_plus
  uv pip install --python "$PYTHON" --no-deps -e "$CHECKOUT" -C editable_mode=compat
}

# robosuite opens the hard-coded /tmp/robosuite.log at import, which on a shared machine may belong to someone else,
# and every import then fails; its macros.py does not consult macros_private.py. The patch of the installed package
# switches the file log off: it applies to the pinned robosuite's bytes (CRLF line endings) and to no other.
patch_robosuite() {
  local inputs site_packages

  inputs=$(cat "$ROBOSUITE_PATCH" <(grep '^robosuite==' "$CONSTRAINTS") | inputs_hash)
  if ! is_made_from "$LIBERO_PLUS_ENV/.pear_inputs_robosuite" "$inputs"; then
    # the package as its wheel ships it, whatever an earlier run left
    uv pip install --python "$PYTHON" --no-deps --reinstall-package robosuite --constraint "$CONSTRAINTS" robosuite
    site_packages=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    patch --fuzz=0 --no-backup-if-mismatch --reject-file=- -p1 -d "$site_packages" < "$ROBOSUITE_PATCH"
    echo "$inputs" > "$LIBERO_PLUS_ENV/.pear_inputs_robosuite"
  fi
}

# LIBERO-plus's scene assets (LIBERO's, and the new objects, lights and textures), which it resolves relative to the
# package: the dataset's archive at its pinned revision, checked and unpacked into the checkout.
install_assets() {
  local archive=$WORK_DIR/cache/libero_plus/assets.zip
  local inputs

  inputs=$(echo "$LIBERO_PLUS_ASSETS_REPO $LIBERO_PLUS_ASSETS_REVISION $LIBERO_PLUS_ASSETS_SHA256" | inputs_hash)
  if ! is_made_from "$LIBERO_PACKAGE/assets/.pear_inputs" "$inputs"; then
    download_from_hub dataset "$LIBERO_PLUS_ASSETS_REPO" "$LIBERO_PLUS_ASSETS_REVISION" "$WORK_DIR/cache/libero_plus" \
      assets.zip
    echo "$LIBERO_PLUS_ASSETS_SHA256  $archive" | sha256sum --check --quiet -
    "$PYTHON" "$INSTALL_DIR/sim/libero_plus/unpack_assets.py" \
      --archive "$archive" --destination "$LIBERO_PACKAGE/assets"
    rm -f "$archive"
    echo "$inputs" > "$LIBERO_PACKAGE/assets/.pear_inputs"
  fi
}

# LIBERO's config inside the environment: without one LIBERO asks for it at its first import and writes the login
# home's ~/.libero.
write_libero_config() {
  mkdir -p "$LIBERO_PLUS_ENV/etc/libero" "$WORK_DIR/datasets/libero"
  cat > "$LIBERO_PLUS_ENV/etc/libero/config.yaml" << YAML
benchmark_root: $LIBERO_PACKAGE
bddl_files: $LIBERO_PACKAGE/bddl_files
init_states: $LIBERO_PACKAGE/init_files
datasets: $WORK_DIR/datasets/libero
assets: $LIBERO_PACKAGE/assets
YAML
}

# One episode's initial observation from a simulator process as a server runs it (the env class's setup_env: Mesa's EGL
# software device, the family's variables, stdin closed); its log: $WORK_DIR/runs/install/libero_plus.
check_initial_observation() {
  LIBERO_MESA_DIR=$ENVS/mesa \
    LIBERO_PYTHON=$PYTHON \
    MAGICK_HOME=$ENVS/imagemagick \
    LIBERO_CONFIG_PATH=$LIBERO_PLUS_ENV/etc/libero \
    PYTHONPATH=$ROOT \
    "$PYTHON" -m pear.sim.checker \
    --tool-config "$ROOT/pear/envs/libero_plus/configs/tool_config.yaml" \
    --env-args "$CHECK_ENV_ARGS" \
    --log-dir "$WORK_DIR/runs/install/libero_plus"
}

require_tools curl getent tar sha256sum git patch
install_uv
install_packages
patch_robosuite
bash "$INSTALL_DIR/sim/mesa/install.sh"
bash "$INSTALL_DIR/sim/imagemagick/install.sh"
install_assets
write_libero_config
check_initial_observation
