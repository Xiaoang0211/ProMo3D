#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUNNER_PYTHON_BIN:-python3}" scripts/_check.py "$@"
