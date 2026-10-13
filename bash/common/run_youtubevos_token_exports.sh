#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ROOT="${ROOT_DIR}/models"
DEFAULT_PYTHON_BIN="${PYTHON_BIN:-}"
if [[ "$#" -gt 0 ]]; then
  MODEL_SELECTIONS=("$@")
else
  MODEL_SELECTIONS=(all)
fi

VGGT_PYTHON_BIN="${VGGT_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/vggt/bin/python}"
VGGT_OMEGA_PYTHON_BIN="${VGGT_OMEGA_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/vggt/bin/python}"
PI3_PYTHON_BIN="${PI3_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/pi3/bin/python}"
DA3_PYTHON_BIN="${DA3_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/depthanything3/bin/python}"
MAPANYTHING_PYTHON_BIN="${MAPANYTHING_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/mapanything/bin/python}"
DUST3R_PYTHON_BIN="${DUST3R_PYTHON_BIN:-/home/xiaoang/anaconda3/envs/dust3r/bin/python}"

YOUTUBEVOS_ROOT="${YOUTUBEVOS_ROOT:-/home/xiaoang/Datasets/Youtube-VOS}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"
TOKEN_ROOT_BASE="${TOKEN_ROOT_BASE:-/media/xiaoang/Extreme SSD}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
VGGT_OUTPUT_DIR="${VGGT_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/vggt_tokens/data/isdynamic}"
VGGT_OMEGA_OUTPUT_DIR="${VGGT_OMEGA_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/vggt_omega_tokens/data/isdynamic}"
PI3X_OUTPUT_DIR="${PI3X_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/pi3x_tokens/data/isdynamic}"
PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/pi3_tokens/data/isdynamic}"
DA3_OUTPUT_DIR="${DA3_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/da3_tokens/data/isdynamic}"
MAPANYTHING_OUTPUT_DIR="${MAPANYTHING_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/mapanything_tokens/data/isdynamic}"
DUST3R_OUTPUT_DIR="${DUST3R_OUTPUT_DIR:-${TOKEN_ROOT_BASE}/dust3r_tokens/data/isdynamic}"
DATASET_NAME="${DATASET_NAME:-youtubevos}"
SPLIT="${SPLIT:-all}"
YOUTUBEVOS_TRAIN_SPLIT="${YOUTUBEVOS_TRAIN_SPLIT:-train_motion}"
YOUTUBEVOS_VAL_SPLIT="${YOUTUBEVOS_VAL_SPLIT:-valid_motion}"
SAVE_MODE="${SAVE_MODE:-sharded_dirs}"
FRAMES_PER_FORWARD="${FRAMES_PER_FORWARD:-8}"
TOKENS_PER_FRAME="${TOKENS_PER_FRAME:-64}"
THRESHOLD="${THRESHOLD:-0.35}"
PREFILL_TEST_FROM_LEGACY="${PREFILL_TEST_FROM_LEGACY:-0}"
FBMS_INFER_ALL_FRAMES="${FBMS_INFER_ALL_FRAMES:-0}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DEVICE="${DEVICE:-cuda}"

VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-${MODEL_ROOT}/vggt/ckpt/model.pt}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${MODEL_ROOT}/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
VGGT_TOKEN_SOURCES="${VGGT_TOKEN_SOURCES:-alt_attn,cam_head,depth_head,point_head}"
VGGT_PREPROCESS_MODE="${VGGT_PREPROCESS_MODE:-native}"
VGGT_OMEGA_TOKEN_SOURCES="${VGGT_OMEGA_TOKEN_SOURCES:-alt_attn,cam_head,depth_head}"
VGGT_OMEGA_PREPROCESS_MODE="${VGGT_OMEGA_PREPROCESS_MODE:-native}"

PI3X_CHECKPOINT="${PI3X_CHECKPOINT:-${MODEL_ROOT}/Pi3/ckpt/model_pi3x.safetensors}"
PI3X_USE_MULTIMODAL_PI3X="${PI3X_USE_MULTIMODAL_PI3X:-0}"
PI3X_TOKEN_SOURCES="${PI3X_TOKEN_SOURCES:-alt_attn,cam_head,conf_head,point_head}"

PI3_CHECKPOINT="${PI3_CHECKPOINT:-${MODEL_ROOT}/Pi3/ckpt/model_pi3.safetensors}"
PI3_TOKEN_SOURCES="${PI3_TOKEN_SOURCES:-alt_attn,cam_head,conf_head,point_head}"

DA3_MODEL_DIR="${DA3_MODEL_DIR:-${MODEL_ROOT}/Depth-Anything-3/src/depth_anything_3/ckpt/DA3-giant-1.1}"
DA3_TOKEN_SOURCES="${DA3_TOKEN_SOURCES:-alt_attn,depth_head}"
DA3_FRAMES_PER_FORWARD="${DA3_FRAMES_PER_FORWARD:-${FRAMES_PER_FORWARD}}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-518}"
DA3_PROCESS_RES_METHOD="${DA3_PROCESS_RES_METHOD:-upper_bound_resize}"
DA3_PREPROCESS_MODE="${DA3_PREPROCESS_MODE:-native}"
DA3_ALLOC_CONF="${DA3_ALLOC_CONF:-expandable_segments:True}"

MAPANYTHING_CHECKPOINT="${MAPANYTHING_CHECKPOINT:-${MODEL_ROOT}/map-anything/ckpt}"
MAPANYTHING_TOKEN_SOURCES="${MAPANYTHING_TOKEN_SOURCES:-inter_frame_block,intra_frame_block,dense_head,pose_head,scale_head}"
MAPANYTHING_PREPROCESS_MODE="${MAPANYTHING_PREPROCESS_MODE:-native}"

DUST3R_WEIGHTS="${DUST3R_WEIGHTS:-${MODEL_ROOT}/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
DUST3R_MODEL_NAME="${DUST3R_MODEL_NAME:-}"
DUST3R_TOKEN_SOURCES="${DUST3R_TOKEN_SOURCES:-enc,dec_view1,dec_view2,head_in_view1,head_in_view2}"
DUST3R_PAIRS_PER_FORWARD="${DUST3R_PAIRS_PER_FORWARD:-8}"
DUST3R_PAIR_STRIDE="${DUST3R_PAIR_STRIDE:-1}"
DUST3R_IMAGE_SIZE="${DUST3R_IMAGE_SIZE:-512}"
DUST3R_SQUARE_OK="${DUST3R_SQUARE_OK:-0}"
DUST3R_PREPROCESS_MODE="${DUST3R_PREPROCESS_MODE:-native}"


usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [model ...]

Models:
  all  vggt  vggt_omega  pi3x  pi3  da3  mapanything  dust3r

Environment variables:
  PYTHON_BIN                 Override python executable for all models
  VGGT_PYTHON_BIN            Default: ${VGGT_PYTHON_BIN}
  VGGT_OMEGA_PYTHON_BIN      Default: ${VGGT_OMEGA_PYTHON_BIN}
  PI3_PYTHON_BIN             Default: ${PI3_PYTHON_BIN}
  DA3_PYTHON_BIN             Default: ${DA3_PYTHON_BIN}
  MAPANYTHING_PYTHON_BIN     Default: ${MAPANYTHING_PYTHON_BIN}
  DUST3R_PYTHON_BIN          Default: ${DUST3R_PYTHON_BIN}
  YOUTUBEVOS_ROOT            Default: ${YOUTUBEVOS_ROOT}
  DAVIS_ROOT                 Default: ${DAVIS_ROOT}
  SEGTRACK_ROOT              Default: ${SEGTRACK_ROOT}
  FBMS_ROOT                  Default: ${FBMS_ROOT}
  TOKEN_ROOT_BASE           Default external SSD token root: ${TOKEN_ROOT_BASE}
  OUTPUT_DIR                 Override output directory for all models
  VGGT_OUTPUT_DIR            Default: ${VGGT_OUTPUT_DIR}
  VGGT_OMEGA_OUTPUT_DIR      Default: ${VGGT_OMEGA_OUTPUT_DIR}
  PI3X_OUTPUT_DIR            Default: ${PI3X_OUTPUT_DIR}
  PI3_OUTPUT_DIR             Default: ${PI3_OUTPUT_DIR}
  DA3_OUTPUT_DIR             Default: ${DA3_OUTPUT_DIR}
  MAPANYTHING_OUTPUT_DIR     Default: ${MAPANYTHING_OUTPUT_DIR}
  DUST3R_OUTPUT_DIR          Default: ${DUST3R_OUTPUT_DIR}
  DATASET_NAME               Default: ${DATASET_NAME}
  SPLIT                      Default: ${SPLIT}
  YOUTUBEVOS_TRAIN_SPLIT     Default: ${YOUTUBEVOS_TRAIN_SPLIT}
  YOUTUBEVOS_VAL_SPLIT       Default: ${YOUTUBEVOS_VAL_SPLIT}
  SAVE_MODE                  Default: ${SAVE_MODE}
  FRAMES_PER_FORWARD         Default: ${FRAMES_PER_FORWARD}
  TOKENS_PER_FRAME           Default: ${TOKENS_PER_FRAME}
  THRESHOLD                  Default: ${THRESHOLD}
  SEED                       Default: ${SEED} (used by dust3r)
  SKIP_EXISTING              Default: ${SKIP_EXISTING}. Set to 0 to rebuild requested split(s) from scratch
  PREFILL_TEST_FROM_LEGACY   Default: ${PREFILL_TEST_FROM_LEGACY}
  FBMS_INFER_ALL_FRAMES      Default: 0. Keep annotated frames only; set to 1 to run FBMS forward on all frames and export only annotated frames

  VGGT_CHECKPOINT            Default: ${VGGT_CHECKPOINT}
  VGGT_TOKEN_SOURCES         Default: ${VGGT_TOKEN_SOURCES}
  VGGT_OMEGA_CHECKPOINT      Default: ${VGGT_OMEGA_CHECKPOINT}
  VGGT_OMEGA_TOKEN_SOURCES   Default: ${VGGT_OMEGA_TOKEN_SOURCES}

  PI3X_CHECKPOINT            Default: ${PI3X_CHECKPOINT}
  PI3X_USE_MULTIMODAL_PI3X   1 to enable multimodal Pi3X branch
  PI3X_TOKEN_SOURCES         Default: ${PI3X_TOKEN_SOURCES}

  PI3_CHECKPOINT             Default: ${PI3_CHECKPOINT}
  PI3_TOKEN_SOURCES          Default: ${PI3_TOKEN_SOURCES}

  DA3_MODEL_DIR              Default: ${DA3_MODEL_DIR}
  DA3_TOKEN_SOURCES          Default: ${DA3_TOKEN_SOURCES}
  DA3_FRAMES_PER_FORWARD     Default: ${DA3_FRAMES_PER_FORWARD}
  DA3_PROCESS_RES            Default: ${DA3_PROCESS_RES}
  DA3_PROCESS_RES_METHOD     Default: ${DA3_PROCESS_RES_METHOD}
  DA3_ALLOC_CONF             Default: ${DA3_ALLOC_CONF}

  MAPANYTHING_CHECKPOINT     Default: ${MAPANYTHING_CHECKPOINT}
  MAPANYTHING_TOKEN_SOURCES  Default: ${MAPANYTHING_TOKEN_SOURCES}

  DUST3R_WEIGHTS             Default: ${DUST3R_WEIGHTS}
  DUST3R_MODEL_NAME          Alternative released model name; leave empty to use local weights
  DUST3R_TOKEN_SOURCES       Default: ${DUST3R_TOKEN_SOURCES}
  DUST3R_PAIRS_PER_FORWARD   Default: ${DUST3R_PAIRS_PER_FORWARD}
  DUST3R_PAIR_STRIDE         Default: ${DUST3R_PAIR_STRIDE}
  DUST3R_IMAGE_SIZE          Default: ${DUST3R_IMAGE_SIZE}
  DUST3R_SQUARE_OK           1 to enable square crops

  MONST3R_WEIGHTS            Default: ${MONST3R_WEIGHTS}
  MONST3R_MODEL_NAME         Alternative released model name; leave empty to use local weights
  MONST3R_TOKEN_SOURCES      Default: ${MONST3R_TOKEN_SOURCES}
  MONST3R_PAIRS_PER_FORWARD  Default: ${MONST3R_PAIRS_PER_FORWARD}
  MONST3R_PAIR_STRIDE        Default: ${MONST3R_PAIR_STRIDE}
  MONST3R_IMAGE_SIZE         Default: ${MONST3R_IMAGE_SIZE}
  MONST3R_SQUARE_OK          1 to enable square crops
USAGE
}

resolve_output_dir() {
  local model_output_dir="$1"
  if [[ -n "${OUTPUT_DIR}" ]]; then
    echo "${OUTPUT_DIR}"
  else
    echo "${model_output_dir}"
  fi
}

log_common() {
  echo "YouTube-VOS root: ${YOUTUBEVOS_ROOT}"
  echo "DAVIS root: ${DAVIS_ROOT}"
  echo "SegTrackv2 root: ${SEGTRACK_ROOT}"
  echo "FBMS-59 root: ${FBMS_ROOT}"
  echo "Split: ${SPLIT}"
  echo "Save mode: ${SAVE_MODE}"
  echo "Frames per forward: ${FRAMES_PER_FORWARD}"
  echo "Tokens per frame: ${TOKENS_PER_FRAME}"
  echo "Threshold: ${THRESHOLD}"
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    echo "FBMS mode: infer all frames, export annotated frames only"
  else
    echo "FBMS mode: infer annotated frames only"
  fi
  echo "Skip existing: ${SKIP_EXISTING}"
}

resolve_python() {
  local model_python="$1"
  if [[ -n "${DEFAULT_PYTHON_BIN}" ]]; then
    echo "${DEFAULT_PYTHON_BIN}"
  else
    echo "${model_python}"
  fi
}

reset_requested_splits_if_needed() {
  local output_dir="$1"
  local layout="${2:-tokens_then_split}"
  if [[ "${SKIP_EXISTING}" != "0" ]]; then
    return
  fi
  if [[ "${SAVE_MODE}" != "sharded_dirs" ]]; then
    return
  fi
  local dataset_root="${output_dir}/${DATASET_NAME}"
  local split_name split_root
  case "${SPLIT}" in
    all)
      for split_name in train val test_davis test_segtrackv2 test_fbms59 test; do
        if [[ "$layout" == "split_then_tokens" ]]; then
        split_root="${dataset_root}/${split_name}/tokens"
      else
        split_root="${dataset_root}/tokens/${split_name}"
      fi
        if [[ -e "${split_root}" ]]; then
          rm -rf "${split_root}"
          echo "[info] removed existing split dir: ${split_root}"
        fi
      done
      ;;
    *)
      if [[ "$layout" == "split_then_tokens" ]]; then
      split_root="${dataset_root}/${SPLIT}/tokens"
    else
      split_root="${dataset_root}/tokens/${SPLIT}"
    fi
      if [[ -e "${split_root}" ]]; then
        rm -rf "${split_root}"
        echo "[info] removed existing split dir: ${split_root}"
      fi
      ;;
  esac
}

prefill_combined_test_from_legacy() {
  local output_dir="$1"
  if [[ "${PREFILL_TEST_FROM_LEGACY}" != "1" ]]; then
    return
  fi
  if [[ "${SAVE_MODE}" != "sharded_dirs" ]]; then
    return
  fi
  if [[ "${SPLIT}" != "test" && "${SPLIT}" != "all" ]]; then
    return
  fi
  local dataset_root="${output_dir}/${DATASET_NAME}"
  local tokens_root="${dataset_root}/tokens"
  local combined_root="${tokens_root}/test"
  local copied=0
  local legacy_split legacy_root seq_path

  mkdir -p "${combined_root}"
  for legacy_split in test_davis test_segtrackv2 test_fbms59; do
    legacy_root="${tokens_root}/${legacy_split}"
    if [[ ! -d "${legacy_root}" ]]; then
      continue
    fi
    while IFS= read -r -d '' seq_path; do
      local seq_name
      seq_name="$(basename "${seq_path}")"
      if [[ -d "${combined_root}/${seq_name}" ]]; then
        continue
      fi
      cp -a "${seq_path}" "${combined_root}/${seq_name}"
      copied=$((copied + 1))
    done < <(find "${legacy_root}" -mindepth 1 -maxdepth 1 -type d -print0)
  done
  if (( copied > 0 )); then
    echo "[info] prefilled ${copied} existing sequences into combined test split: ${combined_root}"
  fi
}

run_vggt() {
  if [[ -z "${VGGT_CHECKPOINT}" ]]; then
    echo "checkpoint is required for vggt" >&2
    exit 1
  fi

  echo "== Running vggt token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${VGGT_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${VGGT_OUTPUT_DIR}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model vggt
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --checkpoint "${VGGT_CHECKPOINT}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${VGGT_TOKEN_SOURCES}"
    --preprocess_mode "${VGGT_PREPROCESS_MODE}"
    --save_mode "${SAVE_MODE}"
    --frames_per_forward "${FRAMES_PER_FORWARD}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --threshold "${THRESHOLD}"
  )
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  "${cmd[@]}"
}

run_vggt_omega() {
  if [[ -z "${VGGT_OMEGA_CHECKPOINT}" ]]; then
    echo "checkpoint is required for vggt_omega" >&2
    exit 1
  fi
  echo "== Running vggt_omega token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${VGGT_OMEGA_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${VGGT_OMEGA_OUTPUT_DIR}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model vggt_omega
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --checkpoint "${VGGT_OMEGA_CHECKPOINT}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${VGGT_OMEGA_TOKEN_SOURCES}"
    --preprocess_mode "${VGGT_OMEGA_PREPROCESS_MODE}"
    --save_mode "${SAVE_MODE}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --frames_per_forward "${FRAMES_PER_FORWARD}"
    --threshold "${THRESHOLD}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  printf '[run] %q ' "${cmd[@]}"; printf '
'
  "${cmd[@]}"
}

run_pi3_variant() {
  local label="$1"
  local model_name="$2"
  local checkpoint="$3"
  local token_sources="$4"
  local model_output_dir="$5"
  local use_multimodal_pi3x="$6"

  if [[ -z "${checkpoint}" ]]; then
    echo "checkpoint is required for ${label}" >&2
    exit 1
  fi

  echo "== Running ${label} token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${PI3_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${model_output_dir}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model "${model_name}"
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --checkpoint "${checkpoint}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${token_sources}"
    --save_mode "${SAVE_MODE}"
    --frames_per_forward "${FRAMES_PER_FORWARD}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --threshold "${THRESHOLD}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  if [[ "${use_multimodal_pi3x}" == "1" ]]; then
    cmd+=(--use_multimodal_pi3x)
  fi
  "${cmd[@]}"
}

run_pi3x() {
  run_pi3_variant "pi3x" "pi3x" "${PI3X_CHECKPOINT}" "${PI3X_TOKEN_SOURCES}" "${PI3X_OUTPUT_DIR}" "${PI3X_USE_MULTIMODAL_PI3X}"
}

run_pi3() {
  run_pi3_variant "pi3" "pi3" "${PI3_CHECKPOINT}" "${PI3_TOKEN_SOURCES}" "${PI3_OUTPUT_DIR}" "0"
}

run_da3() {
  echo "== Running Depth Anything 3 token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${DA3_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${DA3_OUTPUT_DIR}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  echo "DA3 frames per forward: ${DA3_FRAMES_PER_FORWARD}"
  echo "DA3 process_res: ${DA3_PROCESS_RES} (${DA3_PROCESS_RES_METHOD})"
  local alloc_conf
  alloc_conf="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-${DA3_ALLOC_CONF}}}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model depth_anything_3
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --model-dir "${DA3_MODEL_DIR}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${DA3_TOKEN_SOURCES}"
    --preprocess_mode "${DA3_PREPROCESS_MODE}"
    --save_mode "${SAVE_MODE}"
    --frames_per_forward "${DA3_FRAMES_PER_FORWARD}"
    --process_res "${DA3_PROCESS_RES}"
    --process_res_method "${DA3_PROCESS_RES_METHOD}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --threshold "${THRESHOLD}"
  )
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  PYTORCH_ALLOC_CONF="${alloc_conf}" PYTORCH_CUDA_ALLOC_CONF="${alloc_conf}" "${cmd[@]}"
}

append_weights_or_model_name() {
  local mode_name="$1"
  local weights_path="$2"
  local model_name="$3"
  local -n out_ref="$4"
  if [[ -n "${model_name}" ]]; then
    out_ref+=(--model_name "${model_name}")
  else
    if [[ -z "${weights_path}" ]]; then
      echo "${mode_name}: either weights or model_name must be set" >&2
      exit 1
    fi
    out_ref+=(--weights "${weights_path}")
  fi
}

run_mapanything() {
  if [[ -z "${MAPANYTHING_CHECKPOINT}" ]]; then
    echo "MAPANYTHING_CHECKPOINT is required for mapanything" >&2
    exit 1
  fi
  echo "== Running MapAnything token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${MAPANYTHING_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${MAPANYTHING_OUTPUT_DIR}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model map_anything
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --checkpoint "${MAPANYTHING_CHECKPOINT}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${MAPANYTHING_TOKEN_SOURCES}"
    --preprocess_mode "${MAPANYTHING_PREPROCESS_MODE}"
    --save_mode "${SAVE_MODE}"
    --frames_per_forward "${FRAMES_PER_FORWARD}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --threshold "${THRESHOLD}"
  )
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  "${cmd[@]}"
}


run_dust3r() {
  echo "== Running dust3r token export =="
  log_common
  local python_bin output_dir
  python_bin="$(resolve_python "${DUST3R_PYTHON_BIN}")"
  output_dir="$(resolve_output_dir "${DUST3R_OUTPUT_DIR}")"
  reset_requested_splits_if_needed "${output_dir}" "split_then_tokens"
  prefill_combined_test_from_legacy "${output_dir}"
  echo "Python: ${python_bin}"
  echo "Output dir: ${output_dir}"
  local -a cmd=(
    "${python_bin}" -m isdynamic.export.build_youtubevos_isdynamic_npz --model dust3r
    --youtubevos_root "${YOUTUBEVOS_ROOT}"
    --davis_root "${DAVIS_ROOT}"
    --segtrack_root "${SEGTRACK_ROOT}"
    --fbms_root "${FBMS_ROOT}"
    --output_dir "${output_dir}"
    --dataset_name "${DATASET_NAME}"
    --split "${SPLIT}"
    --youtubevos_train_split "${YOUTUBEVOS_TRAIN_SPLIT}"
    --youtubevos_val_split "${YOUTUBEVOS_VAL_SPLIT}"
    --token_sources "${DUST3R_TOKEN_SOURCES}"
    --preprocess_mode "${DUST3R_PREPROCESS_MODE}"
    --save_mode "${SAVE_MODE}"
    --tokens_per_frame "${TOKENS_PER_FRAME}"
    --pairs_per_forward "${DUST3R_PAIRS_PER_FORWARD}"
    --pair_stride "${DUST3R_PAIR_STRIDE}"
    --image_size "${DUST3R_IMAGE_SIZE}"
    --threshold "${THRESHOLD}"
    --seed "${SEED}"
  )
  append_weights_or_model_name "dust3r" "${DUST3R_WEIGHTS}" "${DUST3R_MODEL_NAME}" cmd
  if [[ "${FBMS_INFER_ALL_FRAMES}" == "1" ]]; then
    cmd+=(--fbms_infer_all_frames)
  fi
  if [[ "${DUST3R_SQUARE_OK}" == "1" ]]; then
    cmd+=(--square_ok)
  fi
  "${cmd[@]}"
}

for MODEL_SELECTION in "${MODEL_SELECTIONS[@]}"; do
case "${MODEL_SELECTION}" in
  all)
    run_vggt
    run_vggt_omega
    run_pi3x
    run_pi3
    run_da3
    run_mapanything
    run_dust3r
    ;;
  vggt)
    run_vggt
    ;;
  vggt_omega|vggtomega|omega)
    run_vggt_omega
    ;;
  pi3x)
    run_pi3x
    ;;
  pi3)
    run_pi3
    ;;
  da3|depth-anything-3|depth_anything_3)
    run_da3
    ;;
  mapanything|map-anything)
    run_mapanything
    ;;
  dust3r)
    run_dust3r
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown selection: ${MODEL_SELECTION}" >&2
    usage >&2
    exit 1
    ;;
esac
done
