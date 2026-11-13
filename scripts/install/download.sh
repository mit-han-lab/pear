#!/usr/bin/env bash
# One download of the installer, which every target and program of it uses: a file from an https URL in a bounded
# number of attempts, each within its time limits and to the next of the host's addresses, a failed one printed with
# the address it reached. The caller checks the file against its checksum.
#   bash scripts/install/download.sh <url> <destination>
set -euo pipefail

# the seconds an attempt may take to connect and in all
CONNECT_TIMEOUT_S=20
ATTEMPT_TIMEOUT_S=300
# the attempts: as many as the host with the most addresses has
ATTEMPT_N=6
URL=${1:?the URL}
DESTINATION=${2:?the destination}
HOST=${URL#https://}
HOST=${HOST%%/*}
# the host's addresses: the route from a machine may fail for a part of them, so an attempt does not repeat the last
# one's. None is found behind a proxy, where curl is left to reach the host its own way.
mapfile -t ADDRESSES < <(getent ahostsv4 "$HOST" | cut -d ' ' -f 1 | uniq)

for ((ATTEMPT = 1; ATTEMPT <= ATTEMPT_N; ATTEMPT++)); do
  ADDRESS_ARGS=()
  if [ "${#ADDRESSES[@]}" -gt 0 ]; then
    ADDRESS_ARGS=(--resolve "$HOST:443:${ADDRESSES[(ATTEMPT - 1) % ${#ADDRESSES[@]}]}")
  fi
  if REACHED_ADDRESS=$(curl -fsSL --create-dirs --connect-timeout "$CONNECT_TIMEOUT_S" \
    --max-time "$ATTEMPT_TIMEOUT_S" "${ADDRESS_ARGS[@]}" -o "$DESTINATION" -w '%{remote_ip}' "$URL"); then
    exit 0
  fi
  echo "attempt $ATTEMPT of $ATTEMPT_N: $URL failed at ${REACHED_ADDRESS:-no address}" >&2
done
rm -f "$DESTINATION"
echo "$URL could not be downloaded in $ATTEMPT_N attempts" >&2
exit 1
