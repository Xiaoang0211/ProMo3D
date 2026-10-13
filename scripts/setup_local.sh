#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${RUNNER_PYTHON_BIN:-python3}" "${ROOT_DIR}/scripts/_setup.py" link "$@"
