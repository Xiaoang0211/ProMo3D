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

MODEL="${MODEL:-vggt_omega}"
DATASET="${DATASET:-davis_all}"
SEQUENCE="${SEQUENCE:-breakdance}"
FRAME_INDICES="${FRAME_INDICES:-}"
MAX_STRIP_FRAMES="${MAX_STRIP_FRAMES:-6}"
GROUP="${GROUP:-inter_frame_block}"
BLOCK_IDX="${BLOCK_IDX:-0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
NORM="${NORM:-zscore_l2}"
PCA_DIM="${PCA_DIM:-8}"
NUM_CLUSTERS="${NUM_CLUSTERS:-2}"
SELECTION_RULE="${SELECTION_RULE:-minority}"
SELECTED_CLUSTER="${SELECTED_CLUSTER:-0}"
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

activate_conda_env "$(model_conda_env_name "${MODEL}")" "${PYTHON_BIN_SET_BY_USER}"

cmd=("${PYTHON_BIN}" -m isdynamic.baselines.export_pca_kmeans_paper_style
  --model "${MODEL}"
  --dataset "${DATASET}"
  --sequence "${SEQUENCE}"
  --frame_indices "${FRAME_INDICES}"
  --max_strip_frames "${MAX_STRIP_FRAMES}"
  --group "${GROUP}"
  --block_idx "${BLOCK_IDX}"
  --context_size "${CONTEXT_SIZE}"
  --context_stride "${CONTEXT_STRIDE}"
  --norm "${NORM}"
  --pca_dim "${PCA_DIM}"
  --num_clusters "${NUM_CLUSTERS}"
  --selection_rule "${SELECTION_RULE}"
  --selected_cluster "${SELECTED_CLUSTER}"
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

if [[ -n "${CHECKPOINT}" ]]; then cmd+=(--checkpoint "${CHECKPOINT}"); fi
if [[ -n "${MODEL_DIR}" ]]; then cmd+=(--model_dir "${MODEL_DIR}"); fi
if [[ -n "${WEIGHTS}" ]]; then cmd+=(--weights "${WEIGHTS}"); fi
if [[ -n "${DUST3R_MODEL_NAME}" ]]; then cmd+=(--dust3r_model_name "${DUST3R_MODEL_NAME}"); fi
if [[ -n "${OUT_ROOT}" ]]; then cmd+=(--out_root "${OUT_ROOT}"); fi
if [[ "${USE_MULTIMODAL_PI3X}" == "true" ]]; then cmd+=(--use_multimodal_pi3x); else cmd+=(--no-use_multimodal_pi3x); fi

printf '[run] %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
