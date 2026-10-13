#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES=""
exec "${TEST_PYTHON_BIN:-${ANALYSIS_PYTHON_BIN}}" -m pytest -q "$@"
