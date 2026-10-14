#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${HUB_PYTHON_BIN:-python3}" "$ROOT/scripts/_publish_huggingface.py" "$@"
