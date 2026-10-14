#!/usr/bin/env bash
set -euo pipefail
export PMD_CALLER_CWD="$PWD"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "${RUNNER_PYTHON_BIN:-python3}" scripts/_reproduce.py main predict "$@"
