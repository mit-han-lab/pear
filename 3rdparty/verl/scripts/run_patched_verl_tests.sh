#!/usr/bin/env bash
# Runs verl's own tests in the files that patches/ touch, on the checkout src/, one of verl's tiers at a time.
#   [PYTHON=<the training environment's python>] \
#     bash 3rdparty/verl/scripts/run_patched_verl_tests.sh <cpu | gpu | distributed>
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
TIER=${1:?cpu | gpu | distributed}
PYTHON=${PYTHON:-python}
# prints this repository's faulthandler_timeout: the seconds after which a test still running has its stacks dumped
STACK_DUMP_CODE='import sys, tomllib
print(tomllib.load(open(sys.argv[1], "rb"))["tool"]["pytest"]["ini_options"]["faulthandler_timeout"])'

# The tier of a test file, by verl's naming: cpu the *_on_cpu.py files, distributed the scripts of
# tests/special_distributed (each run by its own torchrun line of verl's run_all.sh, torchrun on PATH), gpu every other
# test file.
tier_of() {
  case "$1" in
    tests/special_distributed/*) echo distributed ;;
    *_on_cpu.py) echo cpu ;;
    *) echo gpu ;;
  esac
}

case "$TIER" in
  cpu | gpu | distributed) ;;
  *)
    echo "unknown tier $TIER (cpu | gpu | distributed)"
    exit 1
    ;;
esac

# from the checkout: inside a repository, `git apply --numstat` names only the files below the working directory.
# Into a variable first: a failure of git ends the script, where a process substitution would yield no test
cd "$PROJECT_DIR/src"
PATCHED_FILES=$(git apply --numstat "$PROJECT_DIR"/patches/*.patch | cut -f 3)

TESTS=()
while IFS= read -r FILE; do
  if [ "$(tier_of "$FILE")" = "$TIER" ]; then
    TESTS+=("$FILE")
  fi
done < <(grep -E '^tests/.*\.py$' <<< "$PATCHED_FILES" | sort -u)
if [ "${#TESTS[@]}" -eq 0 ]; then
  echo "the patches touch no $TIER test of verl"
  exit 0
fi

# verl's tests start Ray clusters of their own: without Ray's usage report, and without its OpenTelemetry metrics
# thread, which ray.shutdown leaves running and which can crash the test process at its exit
export RAY_USAGE_STATS_ENABLED=0 RAY_enable_open_telemetry=0
if [ "$TIER" = distributed ]; then
  for SCRIPT in "${TESTS[@]}"; do
    # verl's own command for the script: its number of ranks is part of the test
    if ! LINE=$(grep -E "^torchrun .* $SCRIPT\$" tests/special_distributed/run_all.sh); then
      echo "verl's run_all.sh has no torchrun line for $SCRIPT"
      exit 1
    fi
    read -r -a COMMAND <<< "$LINE"
    echo "+ ${COMMAND[*]}"
    "${COMMAND[@]}"
  done
else
  # verl's own pytest settings (none) instead of this repository's, whose stack dump alone is taken over
  STACK_DUMP_S=$("$PYTHON" -c "$STACK_DUMP_CODE" "$PROJECT_DIR/../../pyproject.toml")
  "$PYTHON" -m pytest -c /dev/null -o "faulthandler_timeout=$STACK_DUMP_S" --rootdir "$PWD" "${TESTS[@]}"
fi
