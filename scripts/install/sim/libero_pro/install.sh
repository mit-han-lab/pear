#!/usr/bin/env bash
# The LIBERO-PRO simulator environment for the family's server processes, $WORK_DIR/envs/libero_pro: LIBERO-PRO at its
# pinned commit with its perturbed tasks, robosuite and MuJoCo, a CPU-only torch, and Mesa (the target mesa), with
# which the processes render on the CPU. The three LIBERO families are all the package `libero`, so each has its own
# environment and its own checkout.
#   WORK_DIR=<work dir> bash scripts/install.sh libero_pro
set -euo pipefail

# shellcheck source=../../common.sh
source "$(dirname "$0")/../../common.sh"

LIBERO_PRO_ENV=$ENVS/libero_pro
PYTHON=$LIBERO_PRO_ENV/bin/python
CONSTRAINTS=$INSTALL_DIR/sim/libero_pro/constraints.txt
ROBOSUITE_PATCH=$INSTALL_DIR/sim/libero_pro/robosuite_file_log.patch
TASK_CHECKSUMS=$INSTALL_DIR/sim/libero_pro/tasks.sha256
CHECKOUT=$WORK_DIR/src/$(basename "$LIBERO_PRO_URL" .git)
# the libero package inside the checkout, which holds its tasks, initial states and assets
LIBERO_PACKAGE=$CHECKOUT/libero/libero
# the episode of the check
CHECK_ENV_ARGS='{"suite": "libero_spatial", "perturbation": "lan", "task_id": 0, "trial_id": 0}'

# The environment's packages (the extra libero_pro of pyproject.toml) at the versions of its constraints, in one
# resolution, so that numpy<2 (LIBERO, robosuite) holds throughout; then LIBERO-PRO at its pinned commit, editable.
install_packages() {
  create_environment "$LIBERO_PRO_ENV" "$LIBERO_PRO_PYTHON_VERSION"
  if [ ! -d "$CHECKOUT/.git" ]; then
    git clone -q "$LIBERO_PRO_URL" "$CHECKOUT"
  fi
  git -C "$CHECKOUT" fetch -q origin "$LIBERO_PRO_COMMIT"
  git -C "$CHECKOUT" checkout -q "$LIBERO_PRO_COMMIT"
  echo "$(basename "$CHECKOUT"): $(git -C "$CHECKOUT" log --oneline -1)"

  uv pip install --python "$PYTHON" --torch-backend=cpu --constraint "$CONSTRAINTS" \
    -r "$ROOT/pyproject.toml" --extra libero_pro
  uv pip install --python "$PYTHON" --no-deps -e "$CHECKOUT" -C editable_mode=compat
}

# robosuite opens the hard-coded /tmp/robosuite.log at import, which on a shared machine may belong to someone else,
# and every import then fails; its macros.py does not consult macros_private.py. The patch of the installed package
# switches the file log off: it applies to the pinned robosuite's bytes (CRLF line endings) and to no other.
patch_robosuite() {
  local inputs site_packages

  inputs=$(cat "$ROBOSUITE_PATCH" <(grep '^robosuite==' "$CONSTRAINTS") | inputs_hash)
  if ! is_made_from "$LIBERO_PRO_ENV/.pear_inputs_robosuite" "$inputs"; then
    # the package as its wheel ships it, whatever an earlier run left
    uv pip install --python "$PYTHON" --no-deps --reinstall-package robosuite --constraint "$CONSTRAINTS" robosuite
    site_packages=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    patch --fuzz=0 --no-backup-if-mismatch --reject-file=- -p1 -d "$site_packages" < "$ROBOSUITE_PATCH"
    echo "$inputs" > "$LIBERO_PRO_ENV/.pear_inputs_robosuite"
  fi
}

# LIBERO-PRO's perturbed tasks ({suite}_{lan,object,swap,task}: BDDL files and initial states), which its dataset holds
# and its repository does not: the pinned revision, each file checked against tasks.sha256 (the dataset's own list
# covers another set), its task sets placed where LIBERO-PRO's registry looks.
install_tasks() {
  local download=$WORK_DIR/cache/libero_pro
  local inputs

  inputs=$(cat "$TASK_CHECKSUMS" <(echo "$LIBERO_PRO_TASKS_REPO $LIBERO_PRO_TASKS_REVISION") | inputs_hash)
  if ! is_made_from "$LIBERO_PACKAGE/bddl_files/.pear_inputs" "$inputs"; then
    download_from_hub dataset "$LIBERO_PRO_TASKS_REPO" "$LIBERO_PRO_TASKS_REVISION" "$download"
    "$PYTHON" "$INSTALL_DIR/sim/libero_pro/place_tasks.py" \
      --download "$download" --checksums "$TASK_CHECKSUMS" --destination "$LIBERO_PACKAGE"
    echo "$inputs" > "$LIBERO_PACKAGE/bddl_files/.pear_inputs"
  fi
}

# LIBERO's config inside the environment: without one LIBERO asks for it at its first import and writes the login
# home's ~/.libero.
write_libero_config() {
  mkdir -p "$LIBERO_PRO_ENV/etc/libero" "$WORK_DIR/datasets/libero"
  cat > "$LIBERO_PRO_ENV/etc/libero/config.yaml" << YAML
benchmark_root: $LIBERO_PACKAGE
bddl_files: $LIBERO_PACKAGE/bddl_files
init_states: $LIBERO_PACKAGE/init_files
datasets: $WORK_DIR/datasets/libero
assets: $LIBERO_PACKAGE/assets
YAML
}

# One episode's initial observation from a simulator process as a server runs it (the env class's setup_env: Mesa's EGL
# software device, the family's variables, stdin closed); its log: $WORK_DIR/runs/install/libero_pro.
check_initial_observation() {
  LIBERO_MESA_DIR=$ENVS/mesa \
    LIBERO_PYTHON=$PYTHON \
    LIBERO_CONFIG_PATH=$LIBERO_PRO_ENV/etc/libero \
    PYTHONPATH=$ROOT \
    "$PYTHON" -m pear.sim.checker \
    --tool-config "$ROOT/pear/envs/libero_pro/configs/tool_config.yaml" \
    --env-args "$CHECK_ENV_ARGS" \
    --log-dir "$WORK_DIR/runs/install/libero_pro"
}

require_tools curl getent tar sha256sum git patch
install_uv
install_packages
patch_robosuite
bash "$INSTALL_DIR/sim/mesa/install.sh"
install_tasks
write_libero_config
check_initial_observation
