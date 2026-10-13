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
DATASET="${DATASET:-davis_all}"
VARIANT_NAME="${VARIANT_NAME:-pca_kmeans}"
BLOCK_GROUPS="${BLOCK_GROUPS:-}"
GROUP="${GROUP:-}"
BLOCK_MODE="${BLOCK_MODE:-single_block}"
BLOCK_IDX="${BLOCK_IDX:-0}"
BLOCK_INDICES="${BLOCK_INDICES:-}"
CLUSTER_RULE="${CLUSTER_RULE:-minority}"
NORM="${NORM:-zscore_l2}"
PREPROCESS_SCOPE="${PREPROCESS_SCOPE:-frame_group}"
PCA_DIM="${PCA_DIM:-8}"
NUM_CLUSTERS="${NUM_CLUSTERS:-2}"
SEQUENCE="${SEQUENCE:-}"
FRAME_IDX="${FRAME_IDX:--1}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
CONFIG_JSON="${CONFIG_JSON:-}"
CHECKPOINT="${CHECKPOINT:-}"
MODEL_DIR="${MODEL_DIR:-}"
WEIGHTS="${WEIGHTS:-}"
DUST3R_MODEL_NAME="${DUST3R_MODEL_NAME:-}"
YOUTUBEVOS_ROOT="${YOUTUBEVOS_ROOT:-/home/xiaoang/Datasets/Youtube-VOS}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"
OUT_ROOT="${OUT_ROOT:-}"
PROCESS_RES="${PROCESS_RES:-518}"
PROCESS_RES_METHOD="${PROCESS_RES_METHOD:-upper_bound_resize}"
RESOLUTION_SET="${RESOLUTION_SET:-518}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
PREPROCESS_MODE="${PREPROCESS_MODE:-native}"
MAPANYTHING_MACHINE="${MAPANYTHING_MACHINE:-aws}"
USE_MULTIMODAL_PI3X="${USE_MULTIMODAL_PI3X:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

activate_conda_env "$(model_conda_env_name "${MODEL}")" "${PYTHON_BIN_SET_BY_USER}"

cmd=("${PYTHON_BIN}" -m isdynamic.baselines.run_pca_kmeans_motion_masks
  --model "${MODEL}"
  --dataset "${DATASET}"
  --variant_name "${VARIANT_NAME}"
  --sequence "${SEQUENCE}"
  --frame_idx "${FRAME_IDX}"
  --context_size "${CONTEXT_SIZE}"
  --context_stride "${CONTEXT_STRIDE}"
  --cluster_rule "${CLUSTER_RULE}"
  --norm "${NORM}"
  --preprocess_scope "${PREPROCESS_SCOPE}"
  --pca_dim "${PCA_DIM}"
  --num_clusters "${NUM_CLUSTERS}"
  --block_mode "${BLOCK_MODE}"
  --block_idx "${BLOCK_IDX}"
  --anchor_idx 0
  --youtubevos_root "${YOUTUBEVOS_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --segtrack_root "${SEGTRACK_ROOT}"
  --fbms_root "${FBMS_ROOT}"
  --process_res "${PROCESS_RES}"
  --process_res_method "${PROCESS_RES_METHOD}"
  --resolution_set "${RESOLUTION_SET}"
  --image_size "${IMAGE_SIZE}"
  --preprocess_mode "${PREPROCESS_MODE}"
  --mapanything_machine "${MAPANYTHING_MACHINE}")

if [[ -n "${BLOCK_GROUPS}" ]]; then cmd+=(--groups "${BLOCK_GROUPS}"); fi
if [[ -n "${GROUP}" ]]; then cmd+=(--group "${GROUP}"); fi
if [[ -n "${BLOCK_INDICES}" ]]; then cmd+=(--block_indices "${BLOCK_INDICES}"); fi
if [[ -n "${CONFIG_JSON}" ]]; then cmd+=(--config_json "${CONFIG_JSON}"); fi
if [[ -n "${CHECKPOINT}" ]]; then cmd+=(--checkpoint "${CHECKPOINT}"); fi
if [[ -n "${MODEL_DIR}" ]]; then cmd+=(--model_dir "${MODEL_DIR}"); fi
if [[ -n "${WEIGHTS}" ]]; then cmd+=(--weights "${WEIGHTS}"); fi
if [[ -n "${DUST3R_MODEL_NAME}" ]]; then cmd+=(--dust3r_model_name "${DUST3R_MODEL_NAME}"); fi
if [[ -n "${OUT_ROOT}" ]]; then cmd+=(--out_root "${OUT_ROOT}"); fi
if [[ "${USE_MULTIMODAL_PI3X}" == "true" ]]; then cmd+=(--use_multimodal_pi3x); else cmd+=(--no-use_multimodal_pi3x); fi
if [[ "${SKIP_EXISTING}" == "true" ]]; then cmd+=(--skip_existing); else cmd+=(--no-skip_existing); fi

printf '[run] %q ' "${cmd[@]}"
printf '
'
"${cmd[@]}"
