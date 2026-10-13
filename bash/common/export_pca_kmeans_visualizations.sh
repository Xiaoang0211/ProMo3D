#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUTS_DIR="${OUTPUTS_DIR:-${ROOT_DIR}/outputs}"
export OUTPUTS_DIR
export ISDYNAMIC_OUTPUTS_ROOT="${ISDYNAMIC_OUTPUTS_ROOT:-${OUTPUTS_DIR}}"
# shellcheck source=/dev/null
source "${ROOT_DIR}/bash/common/conda_helpers.sh"

PYTHON_BIN_SET_BY_USER=0
if [[ -n "${PYTHON_BIN+x}" ]]; then
  PYTHON_BIN_SET_BY_USER=1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-pi3x}"
DATASET="${DATASET:-davis_all}"
SEQUENCE="${SEQUENCE:-bear}"
FRAME_INDICES="${FRAME_INDICES:-0}"
ANCHOR_VARIANT="${ANCHOR_VARIANT:-pca_kmeans}"
VAL_VARIANT="${VAL_VARIANT:-pca_kmeans_val_selected}"
ORACLE_VARIANT="${ORACLE_VARIANT:-pca_kmeans_oracle}"
PCA_GROUP="${PCA_GROUP:-}"
PMD_GROUP="${PMD_GROUP:-}"
PMD_VARIANT="${PMD_VARIANT:-bilinear_upsample}"
PCA_MASK_ROOT="${PCA_MASK_ROOT:-}"
PMD_MASK_ROOT="${PMD_MASK_ROOT:-${OUTPUTS_DIR}/isdynamic_motion_masks}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"
OUT_DIR="${OUT_DIR:-}"

activate_conda_env "$(model_conda_env_name "${MODEL}")" "${PYTHON_BIN_SET_BY_USER}"
cmd=("${PYTHON_BIN}" -m isdynamic.baselines.export_pca_kmeans_visualizations
  --model "${MODEL}"
  --dataset "${DATASET}"
  --sequence "${SEQUENCE}"
  --frame_indices "${FRAME_INDICES}"
  --anchor_variant "${ANCHOR_VARIANT}"
  --val_variant "${VAL_VARIANT}"
  --oracle_variant "${ORACLE_VARIANT}"
  --pmd_variant "${PMD_VARIANT}"
  --pmd_mask_root "${PMD_MASK_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --segtrack_root "${SEGTRACK_ROOT}"
  --fbms_root "${FBMS_ROOT}")
if [[ -n "${PCA_GROUP}" ]]; then cmd+=(--pca_group "${PCA_GROUP}"); fi
if [[ -n "${PMD_GROUP}" ]]; then cmd+=(--pmd_group "${PMD_GROUP}"); fi
if [[ -n "${PCA_MASK_ROOT}" ]]; then cmd+=(--pca_mask_root "${PCA_MASK_ROOT}"); fi
if [[ -n "${OUT_DIR}" ]]; then cmd+=(--out_dir "${OUT_DIR}"); fi
printf '[run] %q ' "${cmd[@]}"; printf '
'; "${cmd[@]}"
