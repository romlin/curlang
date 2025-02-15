#!/bin/bash
set -e

cd "$(dirname "$0")"

read -p "Are you sure you want to reset? [y/N]: " confirmation

if [[ $confirmation != [yY] ]]; then
  echo -e "Reset canceled."
  exit 1
fi

[[ -d build ]] && rm -r build
[[ -d dist ]] && rm -r dist

[[ -d curlang.egg-info ]] && rm -r curlang.egg-info
[[ -f curlang.sh ]] && rm curlang.sh

echo -e "Reset complete."
