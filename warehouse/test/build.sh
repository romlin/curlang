#!/bin/bash

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "[DEBUG] $(TZ=GMT date +"%Y-%m-%d %H:%M:%S")"
echo -e "build.sh is running in: $SCRIPT_DIR"

export DEFAULT_REPO_NAME=test
export CURLANG_NAME=test

source "$SCRIPT_DIR/device.sh" || {
  echo "Error: Failed to source device.sh" >&2
  exit 1
}

REQUIRED_DEVICES="cpu cuda mps"

if [[ ! $REQUIRED_DEVICES =~ (^|[[:space:]])$DEVICE($|[[:space:]]) ]]; then
  echo "Error: This script requires one of the following devices: $REQUIRED_DEVICES." >&2
  exit 1
fi

source "$SCRIPT_DIR/temp.sh" || {
  echo "Error: Failed to source temp.sh" >&2
  exit 1
}
