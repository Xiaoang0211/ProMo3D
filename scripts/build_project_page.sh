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
# Changed assets receive new URLs even while browsers cache the previous version.
CSS_VERSION="$(sha256sum "$PAGE/style.css")"
JS_VERSION="$(sha256sum "$PAGE/app.js")"
sed -i \
  -e "s|href=\"style.css\"|href=\"style.css?v=${CSS_VERSION:0:12}\"|" \
  -e "s|src=\"app.js\"|src=\"app.js?v=${JS_VERSION:0:12}\"|" \
  "$PAGE/dist/index.html"
cp -R "$PAGE/assets" "$PAGE/dist/"
printf 'Static project page built: %s/dist\n' "$PAGE"
