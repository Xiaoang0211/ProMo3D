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
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"  # 0 means all frames
SEQUENCE_TAGS_CSV="${SEQUENCE_TAGS_CSV:-}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8192}"
VGGT_OFFLOAD_BLOCK_FEATURES_TO_CPU="${VGGT_OFFLOAD_BLOCK_FEATURES_TO_CPU:-1}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
CONTEXT_MODE="${CONTEXT_MODE:-temporal}"
MOTION_MASK_THRESHOLD="${MOTION_MASK_THRESHOLD:-0.5}"
MOTION_MASK_THRESHOLD_MODE="${MOTION_MASK_THRESHOLD_MODE:-otsu}"
MOTION_MASK_THRESHOLD_OFFSET="${MOTION_MASK_THRESHOLD_OFFSET:-0.2}"
PIPELINE_REFINED_AFTER_EACH_SEQUENCE="${PIPELINE_REFINED_AFTER_EACH_SEQUENCE:-0}"

BENCHMARKS="${BENCHMARKS:-davis_all,segtrackv2,fbms59}"
MODELS="${MODELS:-pi3,pi3x,vggt,vggt_omega,depth_anything_3,map_anything,dust3r}"

DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"

# Explicit benchmark split definitions:
DAVIS2016_SPLIT_FILE="${DAVIS2016_SPLIT_FILE:-${DAVIS_ROOT}/ImageSets/2016/val.txt}"
DAVIS2017_VAL_FILE="${DAVIS2017_VAL_FILE:-${DAVIS_ROOT}/ImageSets/2017/val.txt}"
DAVIS2017_TRAIN_FILE="${DAVIS2017_TRAIN_FILE:-${DAVIS_ROOT}/ImageSets/2017/train.txt}"

OUT_DIR="${OUT_DIR:-${OUTPUTS_DIR}/isdynamic_motion_masks}"

PI3_CHECKPOINT="${PI3_CHECKPOINT:-${ROOT_DIR}/models/Pi3/ckpt/model_pi3.safetensors}"
PI3X_CHECKPOINT="${PI3X_CHECKPOINT:-${ROOT_DIR}/models/Pi3/ckpt/model_pi3x.safetensors}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-${ROOT_DIR}/models/vggt/ckpt/model.pt}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${ROOT_DIR}/models/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-${ROOT_DIR}/models/Depth-Anything-3/src/depth_anything_3/ckpt/DA3-giant-1.1}"
MAPANYTHING_CHECKPOINT="${MAPANYTHING_CHECKPOINT:-${ROOT_DIR}/models/map-anything/ckpt}"
DUST3R_WEIGHTS="${DUST3R_WEIGHTS:-${ROOT_DIR}/models/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"

PI3_PROBE_ROOT="${PI3_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/pi3/rank_8_pi3/isdynamic_blockwise_youtubevos_pi3/pair_lowrank_quadratic}"
PI3X_PROBE_ROOT="${PI3X_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/pair_lowrank_quadratic}"
VGGT_PROBE_ROOT="${VGGT_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/vggt/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
VGGT_OMEGA_PROBE_ROOT="${VGGT_OMEGA_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/vggt_omega/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
DA3_PROBE_ROOT="${DA3_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/depth_anything_3/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
MAPANYTHING_PROBE_ROOT="${MAPANYTHING_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/map_anything/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"
DUST3R_PROBE_ROOT="${DUST3R_PROBE_ROOT:-${OUTPUTS_DIR}/model_outputs/dust3r/rank_8/isdynamic_blockwise_youtubevos/pair_lowrank_quadratic}"

PI3_GROUP="${PI3_GROUP:-inter_frame_block}"
PI3X_GROUP="${PI3X_GROUP:-inter_frame_block}"
VGGT_GROUP="${VGGT_GROUP:-inter_frame_block}"
VGGT_OMEGA_GROUP="${VGGT_OMEGA_GROUP:-inter_frame_block}"
DA3_GROUP="${DA3_GROUP:-inter_frame_block}"
MAPANYTHING_GROUP="${MAPANYTHING_GROUP:-inter_frame_block}"
DUST3R_GROUPS="${DUST3R_GROUPS:-dec_view1,head_in_view1}"
MONST3R_GROUPS="${MONST3R_GROUPS:-dec_view1,head_in_view1}"

VGGT_PREPROCESS_MODE="${VGGT_PREPROCESS_MODE:-native}"
VGGT_OMEGA_PREPROCESS_MODE="${VGGT_OMEGA_PREPROCESS_MODE:-native}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-518}"
DA3_PROCESS_RES_METHOD="${DA3_PROCESS_RES_METHOD:-upper_bound_resize}"
DA3_PREPROCESS_MODE="${DA3_PREPROCESS_MODE:-native}"
MAPANYTHING_RESOLUTION_SET="${MAPANYTHING_RESOLUTION_SET:-518}"
MAPANYTHING_PREPROCESS_MODE="${MAPANYTHING_PREPROCESS_MODE:-native}"
DUST3R_IMAGE_SIZE="${DUST3R_IMAGE_SIZE:-512}"
DUST3R_PREPROCESS_MODE="${DUST3R_PREPROCESS_MODE:-native}"
MONST3R_IMAGE_SIZE="${MONST3R_IMAGE_SIZE:-512}"

if [[ "${PIPELINE_REFINED_AFTER_EACH_SEQUENCE}" == "1" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/bash/common/run_refine_motion_masks_all.sh"
fi

trim() {
  echo "$1" | xargs
}

normalize_export_benchmark() {
  case "$1" in
    davis2016|davis2017|davis_all) echo 'davis_all' ;;
    *) echo "$1" ;;
  esac
}

benchmark_requested() {
  local target="$1"
  local raw_benchmark benchmark
  IFS=',' read -r -a requested_benchmarks <<< "${BENCHMARKS}"
  for raw_benchmark in "${requested_benchmarks[@]}"; do
    benchmark="$(trim "${raw_benchmark}")"
    if [[ "${benchmark}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

emit_ordered_benchmarks() {
  local preferred raw_benchmark benchmark normalized
  local -A seen_benchmarks=()
  local -a preferred_order=(davis_all segtrackv2 fbms59)
  for preferred in "${preferred_order[@]}"; do
    for raw_benchmark in "${benchmark_array[@]}"; do
      benchmark="$(trim "${raw_benchmark}")"
      normalized="$(normalize_export_benchmark "${benchmark}")"
      if [[ -n "${benchmark}" && "${normalized}" == "${preferred}" && -z "${seen_benchmarks[${normalized}]:-}" ]]; then
        seen_benchmarks["${normalized}"]=1
        echo "${normalized}"
      fi
    done
  done
}

copy_frame_dir() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
}

sequence_bilinear_complete() {
  local sequence_root="$1"
  local frame_count="$2"
  local frame_idx
  for ((frame_idx=0; frame_idx<frame_count; frame_idx++)); do
    local frame_meta="${sequence_root}/bilinear_upsample/frame_$(printf '%05d' "${frame_idx}")/meta.json"
    if [[ ! -f "${frame_meta}" ]]; then
      return 1
    fi
  done
  return 0
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

print_sequence_specs() {
  local benchmark="$1"
  "${PYTHON_BIN}" - <<PY
from isdynamic.viz.motion_mask_benchmarks import list_sequence_specs, load_sequence_paths_from_spec

benchmark = "${benchmark}"
davis_root = "${DAVIS_ROOT}"
segtrack_root = "${SEGTRACK_ROOT}"
fbms_root = "${FBMS_ROOT}"
requested = {item.strip() for item in "${SEQUENCE_TAGS_CSV}".split(",") if item.strip()}
for spec in list_sequence_specs(benchmark, davis_root=davis_root, segtrack_root=segtrack_root, fbms_root=fbms_root):
    if requested and spec.sequence_tag not in requested:
        continue
    image_paths, _mask_refs, _sequence_tag = load_sequence_paths_from_spec(
        spec,
        davis_root=davis_root,
        segtrack_root=segtrack_root,
        fbms_root=fbms_root,
        fbms_include_unannotated=False,
    )
    fbms_split = spec.fbms_split if spec.fbms_split is not None else "__NONE__"
    print(f"{spec.dataset}\t{spec.sequence}\t{spec.sequence_tag}\t{fbms_split}\t{len(image_paths)}")
PY
}

run_export() {
  local model="$1"
  local benchmark="$2"
  local dataset="$3"
  local sequence="$4"
  local fbms_split="$5"
  local frame_idx="$6"

  local cmd=("${PYTHON_BIN}" -m isdynamic.viz.visualize_motion_mask --model "${model}" --dataset "${dataset}" --benchmark_name "${benchmark}" --sequence "${sequence}" --frame_idx "${frame_idx}" --context_size "${CONTEXT_SIZE}" --context_stride "${CONTEXT_STRIDE}" --device "${DEVICE}" --out_dir "${OUT_DIR}" --threshold "${MOTION_MASK_THRESHOLD}" --threshold_mode "${MOTION_MASK_THRESHOLD_MODE}" --threshold_offset "${MOTION_MASK_THRESHOLD_OFFSET}")
  cmd+=(--davis_root "${DAVIS_ROOT}" --segtrack_root "${SEGTRACK_ROOT}" --fbms_root "${FBMS_ROOT}")
  if [[ -n "${fbms_split}" ]]; then
    cmd+=(--fbms_split "${fbms_split}")
  fi

  case "${model}" in
    pi3)
      cmd+=(--checkpoint "${PI3_CHECKPOINT}" --probe_root "${PI3_PROBE_ROOT}" --group "${PI3_GROUP}" --context_mode "${CONTEXT_MODE}")
      ;;
    pi3x)
      cmd+=(--checkpoint "${PI3X_CHECKPOINT}" --probe_root "${PI3X_PROBE_ROOT}" --group "${PI3X_GROUP}" --context_mode "${CONTEXT_MODE}")
      ;;
    vggt)
      cmd+=(--vggt_checkpoint "${VGGT_CHECKPOINT}" --probe_root "${VGGT_PROBE_ROOT}" --group "${VGGT_GROUP}" --preprocess_mode "${VGGT_PREPROCESS_MODE}")
      if [[ "${VGGT_OFFLOAD_BLOCK_FEATURES_TO_CPU}" == "1" ]]; then
        cmd+=(--offload_block_features_to_cpu)
      else
        cmd+=(--no-offload_block_features_to_cpu)
      fi
      ;;
    vggt_omega)
      cmd+=(--vggt_checkpoint "${VGGT_OMEGA_CHECKPOINT}" --probe_root "${VGGT_OMEGA_PROBE_ROOT}" --group "${VGGT_OMEGA_GROUP}" --preprocess_mode "${VGGT_OMEGA_PREPROCESS_MODE}")
      if [[ "${VGGT_OFFLOAD_BLOCK_FEATURES_TO_CPU}" == "1" ]]; then
        cmd+=(--offload_block_features_to_cpu)
      else
        cmd+=(--no-offload_block_features_to_cpu)
      fi
      ;;
    depth_anything_3|da3)
      cmd+=(--model_dir "${DA3_MODEL_DIR}" --probe_root "${DA3_PROBE_ROOT}" --group "${DA3_GROUP}" --process_res "${DA3_PROCESS_RES}" --process_res_method "${DA3_PROCESS_RES_METHOD}" --preprocess_mode "${DA3_PREPROCESS_MODE}")
      ;;
    map_anything|mapanything)
      cmd+=(--checkpoint "${MAPANYTHING_CHECKPOINT}" --probe_root "${MAPANYTHING_PROBE_ROOT}" --group "${MAPANYTHING_GROUP}" --resolution_set "${MAPANYTHING_RESOLUTION_SET}" --preprocess_mode "${MAPANYTHING_PREPROCESS_MODE}")
      ;;
    dust3r)
      cmd+=(--weights "${DUST3R_WEIGHTS}" --probe_root "${DUST3R_PROBE_ROOT}" --groups "${DUST3R_GROUPS}" --image_size "${DUST3R_IMAGE_SIZE}" --preprocess_mode "${DUST3R_PREPROCESS_MODE}")
      ;;
    *)
      echo "[error] unsupported model: ${model}" >&2
      exit 1
      ;;
  esac

  cmd+=(--batch_size "${SCORE_BATCH_SIZE}")
  "${cmd[@]}"
}

run_export_sequence_batched() {
  local model="$1"
  local benchmark="$2"
  local dataset="$3"
  local sequence="$4"
  local fbms_split="$5"
  local frame_count="$6"
  local feature_batch_size="${FEATURE_BATCH_SIZE}"

  local cmd=("${PYTHON_BIN}" -m isdynamic.viz.visualize_motion_mask --model "${model}" --dataset "${dataset}" --benchmark_name "${benchmark}" --sequence "${sequence}" --frame_start "0" --frame_end "${frame_count}" --frame_step "1" --context_size "${CONTEXT_SIZE}" --context_stride "${CONTEXT_STRIDE}" --device "${DEVICE}" --out_dir "${OUT_DIR}" --threshold "${MOTION_MASK_THRESHOLD}" --threshold_mode "${MOTION_MASK_THRESHOLD_MODE}" --threshold_offset "${MOTION_MASK_THRESHOLD_OFFSET}" --feature_batch_size "${feature_batch_size}")
  cmd+=(--batch_size "${SCORE_BATCH_SIZE}")
  cmd+=(--davis_root "${DAVIS_ROOT}" --segtrack_root "${SEGTRACK_ROOT}" --fbms_root "${FBMS_ROOT}")
  if [[ -n "${fbms_split}" ]]; then
    cmd+=(--fbms_split "${fbms_split}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip_existing)
  fi

  case "${model}" in
    pi3)
      cmd+=(--checkpoint "${PI3_CHECKPOINT}" --probe_root "${PI3_PROBE_ROOT}" --group "${PI3_GROUP}" --context_mode "${CONTEXT_MODE}")
      ;;
    pi3x)
      cmd+=(--checkpoint "${PI3X_CHECKPOINT}" --probe_root "${PI3X_PROBE_ROOT}" --group "${PI3X_GROUP}" --context_mode "${CONTEXT_MODE}")
      ;;
    vggt)
      cmd+=(--vggt_checkpoint "${VGGT_CHECKPOINT}" --probe_root "${VGGT_PROBE_ROOT}" --group "${VGGT_GROUP}" --preprocess_mode "${VGGT_PREPROCESS_MODE}")
      if [[ "${VGGT_OFFLOAD_BLOCK_FEATURES_TO_CPU}" == "1" ]]; then
        cmd+=(--offload_block_features_to_cpu)
      else
        cmd+=(--no-offload_block_features_to_cpu)
      fi
      ;;
    depth_anything_3|da3)
      cmd+=(--model_dir "${DA3_MODEL_DIR}" --probe_root "${DA3_PROBE_ROOT}" --group "${DA3_GROUP}" --process_res "${DA3_PROCESS_RES}" --process_res_method "${DA3_PROCESS_RES_METHOD}" --preprocess_mode "${DA3_PREPROCESS_MODE}")
      ;;
    map_anything|mapanything)
      cmd+=(--checkpoint "${MAPANYTHING_CHECKPOINT}" --probe_root "${MAPANYTHING_PROBE_ROOT}" --group "${MAPANYTHING_GROUP}" --resolution_set "${MAPANYTHING_RESOLUTION_SET}" --preprocess_mode "${MAPANYTHING_PREPROCESS_MODE}")
      ;;
    dust3r)
      cmd+=(--weights "${DUST3R_WEIGHTS}" --probe_root "${DUST3R_PROBE_ROOT}" --groups "${DUST3R_GROUPS}" --image_size "${DUST3R_IMAGE_SIZE}" --preprocess_mode "${DUST3R_PREPROCESS_MODE}")
      ;;
    *)
      echo "[error] unsupported sequence export model: ${model}" >&2
      exit 1
      ;;
  esac

  "${cmd[@]}"
}

maybe_run_refined_after_sequence() {
  local model="$1"
  local benchmark="$2"
  local sequence_tag="$3"
  local group_dir
  local sequence_root
  group_dir="$(group_dir_for_model "${model}")"
  sequence_root="${OUT_DIR}/${model}/${group_dir}/${benchmark}/${sequence_tag}"
  if [[ "${PIPELINE_REFINED_AFTER_EACH_SEQUENCE}" != "1" ]]; then
    return 0
  fi
  run_refined_sequence_variants "${model}" "${group_dir}" "${benchmark}" "${sequence_root}" "${sequence_tag}" || true
  activate_conda_env "$(model_conda_env_name "${model}")" "${PYTHON_BIN_SET_BY_USER}"
}

main() {
  echo "[info] DAVIS export root: davis_all only"
  echo "[info] DAVIS2016 eval split: ${DAVIS2016_SPLIT_FILE}"
  echo "[info] DAVIS2017 eval split: ${DAVIS2017_VAL_FILE}"
  echo "[info] DAVIS-all split: ${DAVIS2017_TRAIN_FILE} + ${DAVIS2017_VAL_FILE}"
  echo "[info] SegTrackv2 root: ${SEGTRACK_ROOT}"
  echo "[info] FBMS-59 root: ${FBMS_ROOT}"
  if [[ -n "${SEQUENCE_TAGS_CSV}" ]]; then
    echo "[info] Sequence subset: ${SEQUENCE_TAGS_CSV}"
  fi

  IFS=',' read -r -a model_array <<< "${MODELS}"
  for raw_model in "${model_array[@]}"; do
    model="$(canonical_model_name "$(trim "${raw_model}")")"
    [[ -n "${model}" ]] || continue
    activate_conda_env "$(model_conda_env_name "${model}")" "${PYTHON_BIN_SET_BY_USER}"
    IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
    while IFS= read -r benchmark; do
      [[ -n "${benchmark}" ]] || continue
      echo "[info] model=${model} benchmark=${benchmark}"
      while IFS=$'\t' read -r dataset sequence sequence_tag fbms_split frame_count; do
        [[ -n "${sequence}" ]] || continue
        if [[ "${fbms_split}" == "__NONE__" ]]; then
          fbms_split=""
        fi
        if [[ "${MAX_FRAMES}" -gt 0 && "${frame_count}" -gt "${MAX_FRAMES}" ]]; then
          frame_count="${MAX_FRAMES}"
        fi
        group_dir="$(group_dir_for_model "${model}")"
        sequence_root="${OUT_DIR}/${model}/${group_dir}/${benchmark}/${sequence_tag}"
        if [[ "${SKIP_EXISTING}" == "1" ]] && sequence_bilinear_complete "${sequence_root}" "${frame_count}"; then
          echo "[skip] ${model} :: ${benchmark} :: ${sequence_tag} :: bilinear_upsample"
          maybe_run_refined_after_sequence "${model}" "${benchmark}" "${sequence_tag}"
          continue
        fi
        if [[ "${model}" != "vggt_omega" ]]; then
          if [[ ( "${benchmark}" == "davis2016" || "${benchmark}" == "davis2017" ) ]] && benchmark_requested "davis_all"; then
            for ((frame_idx=0; frame_idx<frame_count; frame_idx++)); do
              target_out_dir="${sequence_root}/bilinear_upsample/frame_$(printf '%05d' "${frame_idx}")"
              if [[ "${SKIP_EXISTING}" == "1" && -f "${target_out_dir}/meta.json" ]]; then
                echo "[skip] ${model} :: ${benchmark} :: ${sequence_tag} :: frame_idx=${frame_idx}"
                continue
              fi
              source_out_dir="${OUT_DIR}/${model}/${group_dir}/davis_all/${sequence_tag}/bilinear_upsample/frame_$(printf '%05d' "${frame_idx}")"
              if [[ -f "${source_out_dir}/meta.json" ]]; then
                echo "[copy] ${model} :: davis_all -> ${benchmark} :: ${sequence_tag} :: frame_idx=${frame_idx}"
                copy_frame_dir "${source_out_dir}" "${target_out_dir}"
              fi
            done
            maybe_run_refined_after_sequence "${model}" "${benchmark}" "${sequence_tag}"
            continue
          fi
          echo "[run-seq] ${model} :: ${benchmark} :: ${sequence_tag} :: frames=0..$((frame_count-1)) batch=${FEATURE_BATCH_SIZE}"
          run_export_sequence_batched "${model}" "${benchmark}" "${dataset}" "${sequence}" "${fbms_split}" "${frame_count}"
          maybe_run_refined_after_sequence "${model}" "${benchmark}" "${sequence_tag}"
          continue
        fi
        for ((frame_idx=0; frame_idx<frame_count; frame_idx++)); do
          target_out_dir="${sequence_root}/bilinear_upsample/frame_$(printf '%05d' "${frame_idx}")"
          if [[ "${SKIP_EXISTING}" == "1" && -f "${target_out_dir}/meta.json" ]]; then
            echo "[skip] ${model} :: ${benchmark} :: ${sequence_tag} :: frame_idx=${frame_idx}"
            continue
          fi
          if [[ ( "${benchmark}" == "davis2016" || "${benchmark}" == "davis2017" ) ]] && benchmark_requested "davis_all"; then
            source_out_dir="${OUT_DIR}/${model}/${group_dir}/davis_all/${sequence_tag}/bilinear_upsample/frame_$(printf '%05d' "${frame_idx}")"
            if [[ -f "${source_out_dir}/meta.json" ]]; then
              echo "[copy] ${model} :: davis_all -> ${benchmark} :: ${sequence_tag} :: frame_idx=${frame_idx}"
              copy_frame_dir "${source_out_dir}" "${target_out_dir}"
              continue
            fi
          fi
          echo "[run] ${model} :: ${benchmark} :: ${sequence_tag} :: frame_idx=${frame_idx}"
          run_export "${model}" "${benchmark}" "${dataset}" "${sequence}" "${fbms_split}" "${frame_idx}"
        done
        maybe_run_refined_after_sequence "${model}" "${benchmark}" "${sequence_tag}"
      done < <(print_sequence_specs "${benchmark}")
    done < <(emit_ordered_benchmarks)
  done
  return 0
}

main "$@"
