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
MODEL="${MODEL:-pi3x}"
VARIANT_NAME="${VARIANT_NAME:-pca_kmeans}"
GROUP="${GROUP:-}"
BLOCK_GROUPS="${BLOCK_GROUPS:-}"
CONFIG_JSON="${CONFIG_JSON:-}"
MASK_ROOT="${MASK_ROOT:-}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"

activate_conda_env "$(model_conda_env_name "${MODEL}")" "${PYTHON_BIN_SET_BY_USER}"
cmd=("${PYTHON_BIN}" -m isdynamic.baselines.evaluate_pca_kmeans_benchmarks
  --model "${MODEL}"
  --variant_name "${VARIANT_NAME}"
  --davis_root "${DAVIS_ROOT}"
  --segtrack_root "${SEGTRACK_ROOT}"
  --fbms_root "${FBMS_ROOT}")
if [[ -n "${GROUP}" ]]; then cmd+=(--group "${GROUP}"); fi
if [[ -n "${BLOCK_GROUPS}" ]]; then cmd+=(--groups "${BLOCK_GROUPS}"); fi
if [[ -n "${CONFIG_JSON}" ]]; then cmd+=(--config_json "${CONFIG_JSON}"); fi
if [[ -n "${MASK_ROOT}" ]]; then cmd+=(--mask_root "${MASK_ROOT}"); fi
printf '[run] %q ' "${cmd[@]}"; printf '
'; "${cmd[@]}"
