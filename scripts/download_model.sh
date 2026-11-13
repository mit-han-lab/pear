#!/usr/bin/env bash
# Downloads the model a run starts from to $WORK_DIR/models/<name>: the repository at the revision the model's file
# names, from the Hugging Face Hub, with a record of what was fetched; a directory that is what its record states is
# left as it is. On any machine; the installer's target libero_run runs it too.
#   WORK_DIR=<work dir> bash scripts/download_model.sh
#   MODEL_FILE=<a file of examples/models>: the model to download (Qwen3.8-27B's)
set -euo pipefail

# the work directory, uv and the Hub's client are the installer's
# shellcheck source=install/common.sh
source "$(dirname "$0")/install/common.sh"

# the model on the Hugging Face Hub: two plain assignments of its file, which as a whole takes a recipe's settings
MODEL_FILE=${MODEL_FILE:-$ROOT/examples/models/qwen3_8_27b_pear/qwen3_8_27b.sh}
MODEL_REPO=$(sed -n 's/^MODEL_REPO=//p' "$MODEL_FILE")
MODEL_REVISION=$(sed -n 's/^MODEL_REVISION=//p' "$MODEL_FILE")
MODEL_DIR=$WORK_DIR/models/${MODEL_REPO##*/}
# the directory's record of what was fetched
RECORD_NAME=.pear_model

# What the model directory holds, as its record states it: the repository, the revision and every file with its size
# (without the Hub client's own notes under .cache).
model_state() {
  echo "repository $MODEL_REPO"
  echo "revision $MODEL_REVISION"
  (cd "$MODEL_DIR" && find . -type f ! -path './.cache/*' ! -name "$RECORD_NAME" -exec wc -c {} + |
    sed 's/^ *//' | grep -v ' total$' | sort -k 2)
}

if [ -z "$MODEL_REPO" ] || [ -z "$MODEL_REVISION" ]; then
  echo "$MODEL_FILE names no MODEL_REPO and MODEL_REVISION" >&2
  exit 1
fi
if [ -f "$MODEL_DIR/$RECORD_NAME" ]; then
  if STATE_DIFFERENCE=$(diff "$MODEL_DIR/$RECORD_NAME" <(model_state)); then
    exit 0
  fi
  echo "$MODEL_DIR is not what its record states (<: the record, >: the directory) and is fetched anew:"
  echo "$STATE_DIFFERENCE"
  rm -rf "$MODEL_DIR"
fi

require_tools curl getent tar sha256sum
install_uv
# The Hub's client downloads by itself, not by scripts/install/download.sh: it resumes a broken transfer up to five
# times in a row, waits at most ten seconds for a read, and checks every file's size against the Hub's.
download_from_hub model "$MODEL_REPO" "$MODEL_REVISION" "$MODEL_DIR"
model_state > "$MODEL_DIR/$RECORD_NAME"
