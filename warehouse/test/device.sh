#!/bin/bash

for VAR in DEFAULT_REPO_NAME CURLANG_NAME SCRIPT_DIR; do
  if [[ -z "${!VAR}" ]]; then
    echo "Error: $VAR is not set. Please set the $VAR environment variable." >&2
    exit 1
  fi
done

CURRENT_DIR="$(pwd)"
OS=$(uname)
WORK_DIR="$SCRIPT_DIR/$DEFAULT_REPO_NAME"

if [[ "$OS" == "Darwin" ]]; then
  echo "Detected macOS environment"
  DEVICE="mps"
elif [[ "$OS" == "Linux" ]]; then
  echo "Detected Linux environment"
  DEVICE="cpu"
else
  echo "Detected other OS environment"
  DEVICE="cpu"
fi

export VENV_PYTHON="${SCRIPT_DIR}/bin/python"
export VENV_PIP="$(dirname "$VENV_PYTHON")/pip"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "Directory $WORK_DIR does not exist. Creating it..."
  if mkdir -p "$WORK_DIR"; then
    echo "Successfully created directory $WORK_DIR"
  else
    echo "Error: Failed to create directory $WORK_DIR" >&2
    exit 1
  fi
fi

if cd "$WORK_DIR"; then
  echo "Changed to directory $WORK_DIR"
else
  echo "Error: Failed to change to directory $WORK_DIR" >&2
  exit 1
fi
