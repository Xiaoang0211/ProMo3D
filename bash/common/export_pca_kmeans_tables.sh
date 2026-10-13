#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/bash/common/conda_helpers.sh"

PYTHON_BIN_SET_BY_USER=0
if [[ -n "${PYTHON_BIN+x}" ]]; then
  PYTHON_BIN_SET_BY_USER=1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_MODEL="${ENV_MODEL:-pi3x}"
MODELS="${MODELS:-pi3,pi3x,vggt,vggt_omega,depth_anything_3,map_anything,dust3r}"

activate_conda_env "$(model_conda_env_name "${ENV_MODEL}")" "${PYTHON_BIN_SET_BY_USER}"
cmd=("${PYTHON_BIN}" -m isdynamic.baselines.export_pca_kmeans_tables --models "${MODELS}")
printf '[run] %q ' "${cmd[@]}"; printf '
'; "${cmd[@]}"
