#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--assets" ]]; then
  shift
  "${PYTHON_BIN:-python3}" "$ROOT/scripts/build_project_page_assets.py" "$@"
elif [[ $# -gt 0 ]]; then
  printf 'Usage: bash scripts/build_project_page.sh [--assets --archive_outputs /path/to/outputs]\n' >&2
  exit 2
fi
PAGE="$ROOT/project_page"
mkdir -p "$PAGE/dist"
cp "$PAGE/index.html" "$PAGE/style.css" "$PAGE/app.js" "$PAGE/.nojekyll" "$PAGE/dist/"
cp -R "$PAGE/assets" "$PAGE/dist/"
printf 'Static project page built: %s/dist\n' "$PAGE"
