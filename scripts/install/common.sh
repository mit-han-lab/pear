# shellcheck shell=bash disable=SC2034
# Sourced, not run: first by every target of scripts/install.sh, and by scripts/download_model.sh. It installs nothing
# by itself: where the targets install (it makes the work directory), the versions they install
# (scripts/install/versions.env), the settings of every target and the tools the targets share.

if ((BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] < 404)); then
  echo "the installer needs bash 4.4 or newer, this is $BASH_VERSION" >&2
  exit 1
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL_DIR=$ROOT/scripts/install
WORK_DIR=${WORK_DIR:?set WORK_DIR to the work directory (outside the login home)}
# the work directory and the directories directly under it, owner-only where the installer makes them (the work
# directory's parent may be shared with other users); one that exists keeps its mode
# shellcheck disable=SC2174  # the mode is for the directories named, not for the parents -p makes
mkdir -p -m 700 "$WORK_DIR" "$WORK_DIR"/{bin,cache,datasets,envs,models,runs,src,wheels}
ENVS=$WORK_DIR/envs
# shellcheck source=versions.env
source "$INSTALL_DIR/versions.env"
# the machine type (uname -m) the tools' releases and checksums are named by
MACHINE_TYPE=$(uname -m)
if [ "$MACHINE_TYPE" != x86_64 ] && [ "$MACHINE_TYPE" != aarch64 ]; then
  echo "the installer knows x86_64 and aarch64 machines, not $MACHINE_TYPE" >&2
  exit 1
fi

# uv: its binary, the Python builds it downloads, its cache, the directory of its tools, which uvx makes, and its
# credentials store, where it looks when an index answers 404, stay in the work directory (by default the cache, the
# tools and the store are in the login home)
export PATH=$WORK_DIR/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$ENVS/uv-python
export UV_CACHE_DIR=$WORK_DIR/cache/uv
export UV_TOOL_DIR=$WORK_DIR/cache/uv_tools
export UV_CREDENTIALS_DIR=$WORK_DIR/cache/uv_credentials
# two commands at once in one work directory share uv's cache, where one waits for the other's download of a file both
# need: in seconds, longer than uv's own 300, which a large wheel on a slow link takes
export UV_LOCK_TIMEOUT=3600
# pip, which the LMCache build runs: its cache in the work directory too, and no question to PyPI about pip's newest
# version
export PIP_CACHE_DIR=$WORK_DIR/cache/pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
# uv's own Python builds, not the machine's; no project's configuration: run from inside a verl checkout uv would
# apply that pyproject.toml, whose overrides are written for verl's own lock
export UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
# a login shell's compiler modules or Conda base stay out: the builds see the machine's own toolchain only
unset CC CXX FC F77 CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH PYTHONPATH
unset CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
# micromamba, for what conda-forge has and PyPI lacks (the targets mesa, imagemagick and numactl)
MICROMAMBA=$ENVS/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=$ENVS/micromamba/root
# the Hugging Face Hub's command line at the training environment's versions, and its parallel downloads
HF_CLI=(uvx --constraints "$INSTALL_DIR/verl/constraints.txt" --from huggingface_hub hf)
HF_DOWNLOAD_WORKER_N=16

# Ends the target unless the machine has every program named.
require_tools() {
  local tool missing_tools=()

  for tool in "$@"; do
    if ! command -v "$tool" > /dev/null; then
      missing_tools+=("$tool")
    fi
  done
  if [ "${#missing_tools[@]}" -gt 0 ]; then
    echo "missing on $(hostname): ${missing_tools[*]}" >&2
    exit 1
  fi
}

# A file from a URL, kept only when its SHA-256 is the one given: $1 the URL, $2 the checksum, $3 the destination.
download_checked() {
  local url=$1 sha256=$2 destination=$3

  bash "$INSTALL_DIR/download.sh" "$url" "$destination.partial"
  if ! echo "$sha256  $destination.partial" | sha256sum --check --quiet -; then
    rm -f "$destination.partial"
    echo "$url is not the file of SHA-256 $sha256" >&2
    exit 1
  fi
  mv "$destination.partial" "$destination"
}

# The SHA-256 of the standard input, onto which a caller writes what an artifact is made from. The artifact records
# it (is_made_from): made from other inputs than today's, it is made again.
inputs_hash() {
  sha256sum | cut -d ' ' -f 1
}

# Whether the artifact whose record is the file $1 was made from the inputs of hash $2.
is_made_from() {
  [ -f "$1" ] && [ "$(< "$1")" = "$2" ]
}

# uv at its pinned version in $WORK_DIR/bin, from its release archive.
install_uv() {
  local archive=uv-$MACHINE_TYPE-unknown-linux-gnu
  local sha256_name=UV_SHA256_$MACHINE_TYPE

  if [ "$("$WORK_DIR/bin/uv" --version 2> /dev/null | cut -d ' ' -f 2)" != "$UV_VERSION" ]; then
    download_checked "$UV_RELEASES_URL/$UV_VERSION/$archive.tar.gz" "${!sha256_name}" "$WORK_DIR/cache/$archive.tar.gz"
    tar -xzf "$WORK_DIR/cache/$archive.tar.gz" -C "$WORK_DIR/bin" --strip-components=1 "$archive/uv" "$archive/uvx"
    rm "$WORK_DIR/cache/$archive.tar.gz"
  fi
  uv --version
}

# The uv environment $1 on the Python $2: made anew when it is on another Python than the pinned one.
create_environment() {
  local environment=$1 python_version=$2
  local environment_python_version

  environment_python_version=$("$environment/bin/python" -c 'import platform; print(platform.python_version())' \
    2> /dev/null || true)
  if [ "$environment_python_version" != "$python_version" ]; then
    uv venv --clear --python "$python_version" "$environment"
  fi
}

# The CUDA toolkit that an environment's CUDA wheels put into the directory $1, made a complete CUDA_HOME for
# FlashInfer's JIT, torch extensions and source builds: they link -L$CUDA_HOME/lib64 -lcudart, and the wheels ship lib/
# and versioned sonames only.
link_cuda_toolkit() {
  local toolkit=$1
  local library unversioned

  ln -sfn lib "$toolkit/lib64"
  for library in "$toolkit"/lib/lib*.so.[0-9]*; do
    unversioned=${library%%.so.*}.so
    if [ ! -e "$unversioned" ]; then
      ln -s "${library##*/}" "$unversioned"
    fi
  done
}

# micromamba at its pinned version in $ENVS/micromamba, from its release.
install_micromamba() {
  local platform=linux-64
  local sha256_name=MICROMAMBA_SHA256_$MACHINE_TYPE

  if [ "$("$MICROMAMBA" --version 2> /dev/null)" != "$MICROMAMBA_VERSION" ]; then
    if [ "$MACHINE_TYPE" = aarch64 ]; then
      platform=linux-aarch64
    fi
    download_checked "$MICROMAMBA_RELEASES_URL/$MICROMAMBA_VERSION-$MICROMAMBA_BUILD/micromamba-$platform" \
      "${!sha256_name}" "$MICROMAMBA"
    chmod +x "$MICROMAMBA"
  fi
}

# A conda-forge prefix at the versions of its lists, made anew when it was made from other ones: $1 the prefix, then
# the lists. No configuration file of the machine or its user reaches the resolution, and what micromamba keeps under
# a home directory (its repodata cache, its register of environments) stays in the work directory.
install_conda_prefix() {
  local prefix=$1
  local micromamba_home=$WORK_DIR/cache/micromamba
  local inputs list list_args=()

  shift
  inputs=$(cat "$@" <(echo "$CONDA_CHANNEL $MACHINE_TYPE") | inputs_hash)
  if is_made_from "$prefix/.pear_inputs" "$inputs"; then
    return
  fi

  for list in "$@"; do
    list_args+=(--file "$list")
  done
  install_micromamba
  rm -rf "$prefix"
  HOME=$micromamba_home XDG_CACHE_HOME=$micromamba_home/.cache \
    "$MICROMAMBA" create --no-rc -y -q -p "$prefix" -c "$CONDA_CHANNEL" "${list_args[@]}"
  echo "$inputs" > "$prefix/.pear_inputs"
}

# Files of a Hugging Face repository at a revision: $1 model | dataset, $2 the repository, $3 the revision, $4 the
# destination, then the files (all of them when none is named).
download_from_hub() {
  local repo_type=$1 repo=$2 revision=$3 destination=$4

  shift 4
  # the command asks PyPI nothing about its own newest version
  HF_HOME=$WORK_DIR/cache/hf HF_HUB_OFFLINE=0 HF_HUB_DISABLE_UPDATE_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1 \
    "${HF_CLI[@]}" download --repo-type "$repo_type" "$repo" "$@" \
    --revision "$revision" --local-dir "$destination" --max-workers "$HF_DOWNLOAD_WORKER_N" > /dev/null
}
