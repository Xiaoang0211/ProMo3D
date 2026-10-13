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

BENCHMARKS="${BENCHMARKS:-davis2016,davis2017,davis_all,segtrackv2,fbms59}"
MODELS="${MODELS:-pi3,pi3x,vggt,vggt_omega,depth_anything_3,map_anything,dust3r}"
VARIANT="${VARIANT:-bilinear_upsample}"  # bilinear_upsample | upsample_anything | sam2_prompt

MASK_ROOT="${MASK_ROOT:-${OUTPUTS_DIR}/isdynamic_motion_masks}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUTS_DIR}/isdynamic_motion_masks_eval}"

DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"

# Explicit benchmark split definitions:
DAVIS2016_SPLIT_FILE="${DAVIS2016_SPLIT_FILE:-${DAVIS_ROOT}/ImageSets/2016/val.txt}"
DAVIS2017_VAL_FILE="${DAVIS2017_VAL_FILE:-${DAVIS_ROOT}/ImageSets/2017/val.txt}"
DAVIS2017_TRAIN_FILE="${DAVIS2017_TRAIN_FILE:-${DAVIS_ROOT}/ImageSets/2017/train.txt}"

PI3_GROUP="${PI3_GROUP:-inter_frame_block}"
PI3X_GROUP="${PI3X_GROUP:-inter_frame_block}"
VGGT_GROUP="${VGGT_GROUP:-inter_frame_block}"
VGGT_OMEGA_GROUP="${VGGT_OMEGA_GROUP:-inter_frame_block}"
DA3_GROUP="${DA3_GROUP:-inter_frame_block}"
MAPANYTHING_GROUP="${MAPANYTHING_GROUP:-inter_frame_block}"
DUST3R_GROUPS="${DUST3R_GROUPS:-dec_view1,head_in_view1}"
MONST3R_GROUPS="${MONST3R_GROUPS:-dec_view1,head_in_view1}"

trim() {
  echo "$1" | xargs
}

group_dir_for_model() {
  local model="$1"
  case "${model}" in
    pi3) echo "${PI3_GROUP}" ;;
    pi3x) echo "${PI3X_GROUP}" ;;
    vggt) echo "${VGGT_GROUP}" ;;
    vggt_omega) echo "${VGGT_OMEGA_GROUP}" ;;
    depth_anything_3|da3) echo "${DA3_GROUP}" ;;
    map_anything|mapanything) echo "${MAPANYTHING_GROUP}" ;;
    dust3r) echo "${DUST3R_GROUPS//,/__}" ;;
    *) echo "[error] unsupported model: ${model}" >&2; exit 1 ;;
  esac
}

main() {
  echo "[info] DAVIS2016 split: ${DAVIS2016_SPLIT_FILE}"
  echo "[info] DAVIS2017 val split: ${DAVIS2017_VAL_FILE}"
  echo "[info] DAVIS-all split: ${DAVIS2017_TRAIN_FILE} + ${DAVIS2017_VAL_FILE}"
  echo "[info] SegTrackv2 root: ${SEGTRACK_ROOT}"
  echo "[info] FBMS-59 root: ${FBMS_ROOT}"
  echo "[info] variant=${VARIANT}"

  IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
  IFS=',' read -r -a model_array <<< "${MODELS}"
  for raw_model in "${model_array[@]}"; do
    model="$(canonical_model_name "$(trim "${raw_model}")")"
    [[ -n "${model}" ]] || continue
    activate_conda_env "$(model_conda_env_name "${model}")" "${PYTHON_BIN_SET_BY_USER}"
    group_dir="$(group_dir_for_model "${model}")"
    for raw_benchmark in "${benchmark_array[@]}"; do
      benchmark="$(trim "${raw_benchmark}")"
      [[ -n "${benchmark}" ]] || continue
      output_json="${EVAL_ROOT}/${model}/${group_dir}/${VARIANT}/${benchmark}.json"
      echo "[run] ${model} :: ${group_dir} :: ${benchmark} :: ${VARIANT}"
      "${PYTHON_BIN}" -m isdynamic.viz.evaluate_motion_masks \
        --mask_root "${MASK_ROOT}" \
        --model "${model}" \
        --group "${group_dir}" \
        --benchmark "${benchmark}" \
        --variant "${VARIANT}" \
        --davis_root "${DAVIS_ROOT}" \
        --segtrack_root "${SEGTRACK_ROOT}" \
        --fbms_root "${FBMS_ROOT}" \
        --output_json "${output_json}"
    done
  done
}

main "$@"
