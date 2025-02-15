#!/bin/bash
set -e
set -u

echo_stage() {
  printf "\n\033[1;34m>>> %s\033[0m\n" "$1"
}

echo_success() {
  printf "\033[1;32m✓ %s\033[0m\n" "$1"
}

echo_failure() {
  printf "\033[1;31m✗ Error: %s\033[0m\n" "$1"
  exit 1
}

printf "Launching test.sh 🚀\n"

echo_stage "Cleaning up test folder (if it exists)"
if [ -d "test" ]; then
  rm -rf test
  echo_success "Deleted existing test folder"
else
  echo "No existing test folder found"
fi

echo_stage "Checking for Python 3.12 installation"
if ! command -v python3.12 &>/dev/null; then
  echo "Python 3.12 not found. Attempting to install..."

  if command -v brew &>/dev/null; then
    echo "Using Homebrew to install Python 3.12..."
    brew install python@3.12 || echo_failure "Homebrew Python 3.12 installation failed"
  elif command -v apt-get &>/dev/null; then
    echo "Using apt-get to install Python 3.12..."
    sudo apt-get update
    sudo apt-get install -y python3.12 || echo_failure "apt-get Python 3.12 installation failed"
  elif command -v pip3 &>/dev/null; then
    echo "Using pip3 to install Python 3.12..."
    pip3 install python==3.12 || echo_failure "pip3 Python 3.12 installation failed"
  else
    echo_failure "No supported package manager found to install Python 3.12"
  fi
else
  echo_success "Python 3.12 is already installed."
fi

echo_stage "Checking for pipx installation"
if ! command -v pipx &>/dev/null; then
  echo "Pipx not found. Attempting to install..."
  if command -v brew &>/dev/null; then
    echo "Using Homebrew to install pipx..."
    brew install pipx || echo_failure "Homebrew pipx installation failed"
  elif command -v apt-get &>/dev/null; then
    echo "Using apt-get to install pipx..."
    sudo apt-get update
    sudo apt-get install -y pipx || echo_failure "apt-get pipx installation failed"
  elif command -v pip3 &>/dev/null; then
    echo "Using pip3 to install pipx..."
    pip3 install --user pipx || echo_failure "pip3 pipx installation failed"
  else
    echo_failure "No supported package manager found to install pipx"
  fi

  echo "Configuring pipx path..."
  pipx ensurepath || echo_failure "Adding pipx to PATH"

  echo "Enabling global pipx usage..."
  sudo pipx ensurepath --global || echo_failure "Enabling global pipx"

  echo_success "Pipx installed successfully"
else
  echo_success "Pipx is already installed"
fi

echo_stage "Uninstalling existing Curlang (if exists)"
if pipx list | grep -q curlang; then
  pipx uninstall curlang
  echo_success "Existing Curlang uninstalled"
else
  echo "No existing Curlang installation found"
fi

echo_stage "Installing Curlang"
pipx install curlang --python=python3.12 || echo_failure "Curlang installation failed"
echo_success "Curlang installed successfully"

echo_stage "Verifying curlang list"
output=$(curlang list)
if [[ "$output" == *"Curlang Name"* ]]; then
  echo_success "Curlang list command executed successfully"
else
  echo_failure "Curlang list execution failed"
fi

echo_stage "Unboxing test directory"
echo "YES" | curlang unbox test.curlang
echo_success "Curlang unbox test completed"

echo_stage "Checking test folder structure"
if [[ -d "test" && -d "test/build" && $(ls -A "test/build") ]]; then
  echo_success "Test folder and build directory verified"
else
  echo_failure "Test folder or build directory is missing or empty"
fi

echo_stage "Executing curlang build test"
if curlang build test; then
  echo_success "Curlang build test executed successfully"

  if [ -f "test/build/test_pass" ]; then
    echo_success "Test pass file found - Verification complete"
  else
    echo_failure "Test pass file not found"
  fi
else
  echo_failure "Curlang build test execution failed"
fi

if [ -d "test" ]; then
  rm -rf test
fi

printf "\n\033[1;32m🎉 All stages completed successfully!\033[0m\n"
