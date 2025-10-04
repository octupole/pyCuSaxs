#!/usr/bin/env bash
# Bootstrap script to install project dependencies.

set -euo pipefail

# Allow overriding the python executable, default to the one on PATH.
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: cannot find python executable '$PYTHON_BIN'" >&2
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --requirement requirements.txt
