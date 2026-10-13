#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/build_project_page.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/promo3d-pages.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --depth 1 --branch main \
  https://github.com/Xiaoang0211/Xiaoang0211.github.io.git "$WORK/site"
SITE="$WORK/site"

# Replace only this project's generated page, leaving other projects intact.
git -C "$SITE" rm -r --quiet --ignore-unmatch ProMo3D
mkdir -p "$SITE/ProMo3D"
cp -R "$ROOT/project_page/dist/." "$SITE/ProMo3D/"
git -C "$SITE" add -- ProMo3D
if git -C "$SITE" diff --cached --quiet; then
  printf 'Project page is already up to date.\n'
  exit 0
fi
git -C "$SITE" commit -m "Update ProMo3D project page"
git -C "$SITE" push origin HEAD:main
printf 'GitHub Pages build started: https://xiaoang0211.github.io/ProMo3D/\n'
