#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

YOUTUBEVOS_ROOT="${YOUTUBEVOS_ROOT:-/home/xiaoang/Datasets/Youtube-VOS}"
VALIDATION_SPLIT_ROOT="${VALIDATION_SPLIT_ROOT:-${YOUTUBEVOS_ROOT}/valid_motion}"
PROBE_OUTPUTS_ROOT="${PROBE_OUTPUTS_ROOT:-${ROOT_DIR}/outputs/main}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${ROOT_DIR}/outputs/experiments/source_validation}"
VIEW_ROOT="${VIEW_ROOT:-${ARTIFACT_ROOT}/dataset_view}"
MASK_ROOT="${MASK_ROOT:-${ARTIFACT_ROOT}/coarse_masks}"
EVAL_ROOT="${EVAL_ROOT:-${ARTIFACT_ROOT}/eval}"

MODELS="${MODELS:-pi3x,pi3,vggt,depth_anything_3,map_anything,dust3r}"
RUN_STAGES="${RUN_STAGES:-masks,eval}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
DA3_FEATURE_BATCH_SIZE="${DA3_FEATURE_BATCH_SIZE:-1}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-262144}"
DA3_SCORE_BATCH_SIZE="${DA3_SCORE_BATCH_SIZE:-32768}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SEQUENCE_TAGS_CSV="${SEQUENCE_TAGS_CSV:-}"

PI3_PROBE_ROOT="${PI3_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/pi3/rank_8_pi3/isdynamic_blockwise_youtubevos_pi3/pair_lowrank_quadratic}"
PI3X_PROBE_ROOT="${PI3X_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/pair_lowrank_quadratic}"
VGGT_PROBE_ROOT="${VGGT_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/vggt/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
DA3_PROBE_ROOT="${DA3_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/depth_anything_3/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
MAPANYTHING_PROBE_ROOT="${MAPANYTHING_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/map_anything/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
DUST3R_PROBE_ROOT="${DUST3R_PROBE_ROOT:-${PROBE_OUTPUTS_ROOT}/model_outputs/dust3r/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"

trim() {
  echo "$1" | xargs
}

stage_requested() {
  local target="$1"
  local raw stage
  IFS=',' read -r -a stages <<< "${RUN_STAGES}"
  for raw in "${stages[@]}"; do
    stage="$(trim "${raw}")"
    if [[ "${stage}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

prepare_validation_view() {
  local image_root="${VALIDATION_SPLIT_ROOT}/JEPGImages"
  if [[ ! -d "${image_root}" ]]; then
    image_root="${VALIDATION_SPLIT_ROOT}/JPEGImages"
  fi
  local annotation_root="${VALIDATION_SPLIT_ROOT}/Annotations"
  if [[ ! -d "${image_root}" || ! -d "${annotation_root}" ]]; then
    echo "[error] invalid YouTube-VOS validation split: ${VALIDATION_SPLIT_ROOT}" >&2
    exit 1
  fi

  mkdir -p \
    "${VIEW_ROOT}/ImageSets/2016" \
    "${VIEW_ROOT}/ImageSets/2017" \
    "${VIEW_ROOT}/JPEGImages" \
    "${VIEW_ROOT}/Annotations"
  ln -sfn "${image_root}" "${VIEW_ROOT}/JPEGImages/480p"
  ln -sfn "${annotation_root}" "${VIEW_ROOT}/Annotations/480p"
  : > "${VIEW_ROOT}/ImageSets/2016/val.txt"
  : > "${VIEW_ROOT}/ImageSets/2017/train.txt"
  find "${image_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | sort > "${VIEW_ROOT}/ImageSets/2017/val.txt"

  local sequence_count
  local frame_count
  sequence_count="$(wc -l < "${VIEW_ROOT}/ImageSets/2017/val.txt")"
  frame_count="$(find "${image_root}" -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)"
  echo "[info] YouTube-VOS validation view: ${sequence_count} sequences, ${frame_count} frames"
}

run_masks() {
  local raw_model model feature_batch_size score_batch_size
  IFS=',' read -r -a models <<< "${MODELS}"
  for raw_model in "${models[@]}"; do
    model="$(trim "${raw_model}")"
    [[ -n "${model}" ]] || continue
    score_batch_size="${SCORE_BATCH_SIZE}"
    if [[ "${model}" == "da3" || "${model}" == "depth_anything_3" ]]; then
      score_batch_size="${DA3_SCORE_BATCH_SIZE}"
      feature_batch_size="${DA3_FEATURE_BATCH_SIZE}"
    else
      feature_batch_size="${FEATURE_BATCH_SIZE}"
    fi
    OUTPUTS_DIR="${PROBE_OUTPUTS_ROOT}" \
    OUT_DIR="${MASK_ROOT}" \
    DAVIS_ROOT="${VIEW_ROOT}" \
    BENCHMARKS=davis_all \
    MODELS="${model}" \
    CONTEXT_SIZE="${CONTEXT_SIZE}" \
    CONTEXT_STRIDE="${CONTEXT_STRIDE}" \
    FEATURE_BATCH_SIZE="${feature_batch_size}" \
    SCORE_BATCH_SIZE="${score_batch_size}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    MAX_FRAMES="${MAX_FRAMES}" \
    SEQUENCE_TAGS_CSV="${SEQUENCE_TAGS_CSV}" \
    ISDYNAMIC_MASK_OUTPUT_MODE=metrics_only \
    MOTION_MASK_THRESHOLD=0.5 \
    MOTION_MASK_THRESHOLD_MODE=otsu \
    MOTION_MASK_THRESHOLD_OFFSET=0.2 \
    PI3_PROBE_ROOT="${PI3_PROBE_ROOT}" \
    PI3X_PROBE_ROOT="${PI3X_PROBE_ROOT}" \
    VGGT_PROBE_ROOT="${VGGT_PROBE_ROOT}" \
    DA3_PROBE_ROOT="${DA3_PROBE_ROOT}" \
    MAPANYTHING_PROBE_ROOT="${MAPANYTHING_PROBE_ROOT}" \
    DUST3R_PROBE_ROOT="${DUST3R_PROBE_ROOT}" \
    bash "${ROOT_DIR}/bash/common/run_export_motion_masks_all.sh"
  done
}

run_eval() {
  OUTPUTS_DIR="${PROBE_OUTPUTS_ROOT}" \
  MASK_ROOT="${MASK_ROOT}" \
  EVAL_ROOT="${EVAL_ROOT}" \
  DAVIS_ROOT="${VIEW_ROOT}" \
  BENCHMARKS=davis_all \
  MODELS="${MODELS}" \
  VARIANT=bilinear_upsample \
  bash "${ROOT_DIR}/bash/common/run_eval_motion_masks_all.sh"
}

main() {
  prepare_validation_view
  if stage_requested masks; then
    run_masks
  fi
  if stage_requested eval; then
    run_eval
  fi
}

main "$@"
