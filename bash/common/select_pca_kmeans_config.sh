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
CHECKPOINT="${CHECKPOINT:-}"
MODEL_DIR="${MODEL_DIR:-}"
WEIGHTS="${WEIGHTS:-}"
DUST3R_MODEL_NAME="${DUST3R_MODEL_NAME:-}"
YOUTUBEVOS_ROOT="${YOUTUBEVOS_ROOT:-/home/xiaoang/Datasets/Youtube-VOS}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
PROCESS_RES="${PROCESS_RES:-518}"
PROCESS_RES_METHOD="${PROCESS_RES_METHOD:-upper_bound_resize}"
RESOLUTION_SET="${RESOLUTION_SET:-518}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
PREPROCESS_MODE="${PREPROCESS_MODE:-native}"
MAPANYTHING_MACHINE="${MAPANYTHING_MACHINE:-aws}"
USE_MULTIMODAL_PI3X="${USE_MULTIMODAL_PI3X:-false}"

activate_conda_env "$(model_conda_env_name "${MODEL}")" "${PYTHON_BIN_SET_BY_USER}"
cmd=("${PYTHON_BIN}" -m isdynamic.baselines.select_pca_kmeans_config
  --model "${MODEL}"
  --youtubevos_root "${YOUTUBEVOS_ROOT}"
  --davis_root "${DAVIS_ROOT}"
  --segtrack_root "${SEGTRACK_ROOT}"
  --fbms_root "${FBMS_ROOT}"
  --max_sequences "${MAX_SEQUENCES}"
  --max_frames_per_sequence "${MAX_FRAMES_PER_SEQUENCE}"
  --context_size "${CONTEXT_SIZE}"
  --context_stride "${CONTEXT_STRIDE}"
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
if [[ "${USE_MULTIMODAL_PI3X}" == "true" ]]; then cmd+=(--use_multimodal_pi3x); else cmd+=(--no-use_multimodal_pi3x); fi
printf '[run] %q ' "${cmd[@]}"; printf '
'; "${cmd[@]}"
