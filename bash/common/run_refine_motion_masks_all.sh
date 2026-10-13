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
VARIANTS="${VARIANTS:-sam2_prompt}"
MODELS="${MODELS:-pi3,pi3x,vggt,vggt_omega,depth_anything_3,map_anything,dust3r}"
BENCHMARKS="${BENCHMARKS:-davis_all,segtrackv2,fbms59}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

MASK_ROOT="${MASK_ROOT:-${OUTPUTS_DIR}/isdynamic_motion_masks}"

PI3_GROUP="${PI3_GROUP:-inter_frame_block}"
PI3X_GROUP="${PI3X_GROUP:-inter_frame_block}"
VGGT_GROUP="${VGGT_GROUP:-inter_frame_block}"
VGGT_OMEGA_GROUP="${VGGT_OMEGA_GROUP:-inter_frame_block}"
DA3_GROUP="${DA3_GROUP:-inter_frame_block}"
MAPANYTHING_GROUP="${MAPANYTHING_GROUP:-inter_frame_block}"
DUST3R_GROUPS="${DUST3R_GROUPS:-dec_view1,head_in_view1}"
MONST3R_GROUPS="${MONST3R_GROUPS:-dec_view1,head_in_view1}"

SAM2_REPO_ROOT="${SAM2_REPO_ROOT:-${ROOT_DIR}/sam2}"
SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-checkpoints/sam2.1_hiera_large.pt}"
SAM2_DEVICE="${SAM2_DEVICE:-cuda}"
SAM2_OBJECT_ID="${SAM2_OBJECT_ID:-1}"
SAM2_SCORE_THRESH="${SAM2_SCORE_THRESH:-0.0}"
SAM2_PROMPT_SCORE_SOURCE="${SAM2_PROMPT_SCORE_SOURCE:-upsampled}"
SAM2_PROMPT_THRESHOLD="${SAM2_PROMPT_THRESHOLD:-0.9}"
SAM2_PROMPT_THRESHOLD_MODE="${SAM2_PROMPT_THRESHOLD_MODE:-otsu}"
SAM2_PROMPT_THRESHOLD_OFFSET="${SAM2_PROMPT_THRESHOLD_OFFSET:-0.2}"
SAM2_PROMPT_ERODE_ITERATIONS="${SAM2_PROMPT_ERODE_ITERATIONS:-2}"
SAM2_INSTANCE_MEAN_SCORE_THRESHOLD="${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD:-0.0}"
SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_MODE="${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_MODE:-otsu}"
SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET="${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET:-0.3}"
SAM2_PROPAGATE_MODE="${SAM2_PROPAGATE_MODE:-parity}"
SAM2_MAX_AREA_INCREASE="${SAM2_MAX_AREA_INCREASE:-}"
SAM2_WINDOW_SIZE="${SAM2_WINDOW_SIZE:-30}"
SAM2_WINDOW_OVERLAP="${SAM2_WINDOW_OVERLAP:-0}"
SAM2_OFFLOAD_VIDEO_TO_CPU="${SAM2_OFFLOAD_VIDEO_TO_CPU:-0}"
SAM2_OFFLOAD_STATE_TO_CPU="${SAM2_OFFLOAD_STATE_TO_CPU:-0}"
SAM2_ASYNC_LOADING_FRAMES="${SAM2_ASYNC_LOADING_FRAMES:-0}"
SAM2_CONDA_ENV="${SAM2_CONDA_ENV:-pi3}"

UPA_REPO_ROOT="${UPA_REPO_ROOT:-${ROOT_DIR}/Upsample-Anything_Pytorch}"
UPA_HEATMAP_SOURCE="${UPA_HEATMAP_SOURCE:-patch}"
UPA_THRESHOLD="${UPA_THRESHOLD:-0.5}"
UPA_THRESHOLD_MODE="${UPA_THRESHOLD_MODE:-fixed}"
UPA_THRESHOLD_OFFSET="${UPA_THRESHOLD_OFFSET:-0.0}"
UPA_MIN_COMPONENT_AREA="${UPA_MIN_COMPONENT_AREA:-0}"
UPA_MAX_HOLE_AREA="${UPA_MAX_HOLE_AREA:-0}"
UPA_INSTANCE_MEAN_SCORE_THRESHOLD="${UPA_INSTANCE_MEAN_SCORE_THRESHOLD:-0.0}"
UPA_CONDA_ENV="${UPA_CONDA_ENV:-pi3}"

trim() {
  echo "$1" | xargs
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
  local preferred raw_benchmark benchmark
  local -a preferred_order=(davis_all davis2016 davis2017 segtrackv2 fbms59)
  for preferred in "${preferred_order[@]}"; do
    for raw_benchmark in "${benchmark_array[@]}"; do
      benchmark="$(trim "${raw_benchmark}")"
      if [[ -n "${benchmark}" && "${benchmark}" == "${preferred}" ]]; then
        echo "${benchmark}"
      fi
    done
  done
}

copy_variant_dir() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
}

variant_sequence_complete() {
  local direct_root="$1"
  local output_root="$2"
  local variant="$3"
  local frame_dir frame_name

  [[ -f "${output_root}/meta.json" ]] || return 1

  shopt -s nullglob
  for frame_dir in "${direct_root}"/frame_*; do
    [[ -d "${frame_dir}" ]] || continue
    frame_name="$(basename "${frame_dir}")"
    case "${variant}" in
      sam2_prompt)
        if [[ ! -f "${output_root}/${frame_name}.npy" && ! -f "${output_root}/${frame_name}.png" ]]; then
          shopt -u nullglob
          return 1
        fi
        ;;
      upsample_anything)
        if [[ ! -f "${output_root}/${frame_name}/meta.json" ]]; then
          shopt -u nullglob
          return 1
        fi
        ;;
      *)
        shopt -u nullglob
        return 1
        ;;
    esac
  done
  shopt -u nullglob
  return 0
}

resolve_sam2_repo_root() {
  local requested="${SAM2_REPO_ROOT}"
  local builtin="${ROOT_DIR}/sam2"
  if [[ -d "${requested}" ]]; then
    echo "${requested}"
    return
  fi
  if [[ -d "${builtin}" ]]; then
    echo "${builtin}"
    return
  fi
  echo "${requested}"
}

resolve_sam2_path() {
  local repo_root="$1"
  local requested="$2"
  local fallback="$3"

  if [[ "${requested}" = /* && -f "${requested}" ]]; then
    echo "${requested}"
    return
  fi
  if [[ -f "${repo_root}/${requested}" ]]; then
    echo "${requested}"
    return
  fi
  if [[ -f "${repo_root}/sam2/${requested}" ]]; then
    echo "sam2/${requested}"
    return
  fi
  if [[ -f "${repo_root}/${fallback}" ]]; then
    echo "${fallback}"
    return
  fi
  echo "${requested}"
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

sequence_roots_for_scope() {
  local model="$1"
  local benchmark="$2"
  local group_dir
  group_dir="$(group_dir_for_model "${model}")"
  local root="${MASK_ROOT}/${model}/${group_dir}/${benchmark}"
  if [[ ! -d "${root}" ]]; then
    return
  fi
  find "${root}" -mindepth 1 -maxdepth 1 -type d | sort
}

resolve_video_path() {
  local direct_root="$1"
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

direct_root = Path("${direct_root}")
frame_dirs = sorted([path for path in direct_root.iterdir() if path.is_dir() and path.name.startswith("frame_")])
if not frame_dirs:
    raise SystemExit(1)
meta_path = frame_dirs[0] / "meta.json"
with meta_path.open("r", encoding="utf-8") as handle:
    meta = json.load(handle)
image_path = meta.get("image_path")
if not isinstance(image_path, str):
    raise SystemExit(1)
print(str(Path(image_path).resolve().parent))
PY
}

run_sam2_sequence() {
  local direct_root="$1"
  local output_root="$2"
  local variant_name="$3"
  local video_path
  video_path="$(resolve_video_path "${direct_root}")"
  local sam2_repo_root
  sam2_repo_root="$(resolve_sam2_repo_root)"
  local sam2_cfg
  sam2_cfg="$(resolve_sam2_path "${sam2_repo_root}" "${SAM2_CFG}" "sam2/configs/sam2.1/sam2.1_hiera_l.yaml")"
  local sam2_checkpoint
  sam2_checkpoint="$(resolve_sam2_path "${sam2_repo_root}" "${SAM2_CHECKPOINT}" "checkpoints/sam2.1_hiera_large.pt")"
  local -a cmd
  if [[ -n "${SAM2_PYTHON_BIN:-}" ]]; then
    cmd=("${SAM2_PYTHON_BIN}" -m isdynamic.viz.refine_motion_masks_with_sam2)
  else
    ensure_conda_loaded
    cmd=(conda run --no-capture-output -n "${SAM2_CONDA_ENV}" python -m isdynamic.viz.refine_motion_masks_with_sam2)
  fi
  cmd+=(
    --mask_root "${direct_root}"
    --video_path "${video_path}"
    --sam2_repo_root "${sam2_repo_root}"
    --sam2_cfg "${sam2_cfg}"
    --sam2_checkpoint "${sam2_checkpoint}"
    --output_dir "${output_root}"
    --variant_name "${variant_name}"
    --score_thresh "${SAM2_SCORE_THRESH}"
    --prompt_score_source "${SAM2_PROMPT_SCORE_SOURCE}"
    --prompt_threshold "${SAM2_PROMPT_THRESHOLD}"
    --prompt_threshold_mode "${SAM2_PROMPT_THRESHOLD_MODE}"
    --prompt_threshold_offset "${SAM2_PROMPT_THRESHOLD_OFFSET}"
    --prompt_erode_iterations "${SAM2_PROMPT_ERODE_ITERATIONS}"
    --instance_mean_score_threshold "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD}"
    --instance_mean_score_threshold_mode "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_MODE}"
    --instance_mean_score_threshold_offset "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}"
    --propagate_mode "${SAM2_PROPAGATE_MODE}"
    --window_size "${SAM2_WINDOW_SIZE}"
    --window_overlap "${SAM2_WINDOW_OVERLAP}"
    --device "${SAM2_DEVICE}"
  )
  if [[ -n "${SAM2_MAX_AREA_INCREASE}" ]]; then
    cmd+=(--max_area_increase "${SAM2_MAX_AREA_INCREASE}")
  fi
  if [[ "${SAM2_OFFLOAD_VIDEO_TO_CPU}" == "1" ]]; then
    cmd+=(--offload_video_to_cpu)
  fi
  if [[ "${SAM2_OFFLOAD_STATE_TO_CPU}" == "1" ]]; then
    cmd+=(--offload_state_to_cpu)
  fi
  if [[ "${SAM2_ASYNC_LOADING_FRAMES}" == "1" ]]; then
    cmd+=(--async_loading_frames)
  fi
  "${cmd[@]}"
}

run_upa_sequence() {
  local direct_root="$1"
  local output_root="$2"
  local variant_name="$3"
  ensure_conda_loaded
  local -a cmd=(
    conda run --no-capture-output -n "${UPA_CONDA_ENV}" python -m isdynamic.viz.refine_motion_masks_with_upsample_anything
  )
  cmd+=(
    --mask_root "${direct_root}"
    --output_dir "${output_root}"
    --variant_name "${variant_name}"
    --upa_repo_root "${UPA_REPO_ROOT}"
    --heatmap_source "${UPA_HEATMAP_SOURCE}"
    --threshold "${UPA_THRESHOLD}"
    --threshold_mode "${UPA_THRESHOLD_MODE}"
    --threshold_offset "${UPA_THRESHOLD_OFFSET}"
    --min_component_area "${UPA_MIN_COMPONENT_AREA}"
    --max_hole_area "${UPA_MAX_HOLE_AREA}"
    --instance_mean_score_threshold "${UPA_INSTANCE_MEAN_SCORE_THRESHOLD}"
  )
  "${cmd[@]}"
}

resolve_variant_output_root() {
  local sequence_root="$1"
  local variant="$2"
  local resolved_variant
  resolved_variant="$(resolve_variant_name "${variant}")"
  echo "${sequence_root}/${resolved_variant}"
}

resolve_variant_name() {
  local variant="$1"
  local resolved="${variant}"
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_PROMPT_THRESHOLD_MODE}" == "fixed" ]]; then
    resolved="sam2_prompt_fixed_${SAM2_PROMPT_THRESHOLD}"
  fi
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_PROMPT_THRESHOLD_MODE}" == "otsu" && "${SAM2_PROMPT_THRESHOLD_OFFSET}" != "0.0" && "${SAM2_PROMPT_THRESHOLD_OFFSET}" != "0" ]]; then
    resolved="${resolved}_otsu_off${SAM2_PROMPT_THRESHOLD_OFFSET}"
  fi
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_PROMPT_ERODE_ITERATIONS}" != "0" ]]; then
    resolved="${resolved}_erode${SAM2_PROMPT_ERODE_ITERATIONS}"
  fi
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_WINDOW_SIZE}" != "0" ]]; then
    resolved="${resolved}_w${SAM2_WINDOW_SIZE}"
    if [[ "${SAM2_WINDOW_OVERLAP}" != "0.0" && "${SAM2_WINDOW_OVERLAP}" != "0" ]]; then
      resolved="${resolved}_ov${SAM2_WINDOW_OVERLAP}"
    fi
  fi
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_PROPAGATE_MODE}" == "single" ]]; then
    resolved="${resolved}_single"
  fi
  if [[ "${variant}" == "sam2_prompt" && "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_MODE}" == "otsu" ]]; then
    resolved="${resolved}_mean_otsu"
    if [[ "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}" != "0.0" && "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}" != "0" ]]; then
      resolved="${resolved}_off${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}"
    fi
  elif [[ "${variant}" == "sam2_prompt" && "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD}" != "0.0" && "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD}" != "0" ]]; then
    resolved="${resolved}_mean${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD}"
    if [[ "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}" != "0.0" && "${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}" != "0" ]]; then
      resolved="${resolved}_off${SAM2_INSTANCE_MEAN_SCORE_THRESHOLD_OFFSET}"
    fi
  fi
  if [[ "${variant}" == "upsample_anything" && "${UPA_HEATMAP_SOURCE}" == "upsampled" ]]; then
    resolved="upsample_anything_from_upsampled"
  fi
  if [[ "${variant}" == "upsample_anything" && "${UPA_THRESHOLD_MODE}" == "otsu" && "${UPA_THRESHOLD_OFFSET}" != "0.0" && "${UPA_THRESHOLD_OFFSET}" != "0" ]]; then
    resolved="${resolved}_otsu_off${UPA_THRESHOLD_OFFSET}"
  fi
  if [[ "${variant}" == "upsample_anything" && "${UPA_INSTANCE_MEAN_SCORE_THRESHOLD}" != "0.0" && "${UPA_INSTANCE_MEAN_SCORE_THRESHOLD}" != "0" ]]; then
    resolved="${resolved}_mean${UPA_INSTANCE_MEAN_SCORE_THRESHOLD}"
  fi
  echo "${resolved}"
}

run_refined_sequence_variants() {
  local model="$1"
  local group_dir="$2"
  local benchmark="$3"
  local sequence_root="$4"
  local sequence_tag="$5"
  local ran_local=0

  IFS=',' read -r -a requested_variants <<< "${VARIANTS}"
  local ordered_variants=()
  local raw_variant
  local variant
  for variant in sam2_prompt upsample_anything; do
    for raw_variant in "${requested_variants[@]}"; do
      if [[ "$(trim "${raw_variant}")" == "${variant}" ]]; then
        ordered_variants+=("${variant}")
        break
      fi
    done
  done
  direct_root="${sequence_root}/bilinear_upsample"
  [[ -d "${direct_root}" ]] || return 0

  for variant in "${ordered_variants[@]}"; do
    local output_root
    local variant_name
    local source_root
    output_root="$(resolve_variant_output_root "${sequence_root}" "${variant}")"
    variant_name="$(resolve_variant_name "${variant}")"
    if [[ "${SKIP_EXISTING}" == "1" ]] && variant_sequence_complete "${direct_root}" "${output_root}" "${variant}"; then
      echo "[skip] ${model} :: ${group_dir} :: ${benchmark} :: ${sequence_tag} :: ${variant}"
      continue
    fi
    if [[ ( "${benchmark}" == "davis2016" || "${benchmark}" == "davis2017" ) ]] && benchmark_requested "davis_all"; then
      source_root="${MASK_ROOT}/${model}/${group_dir}/davis_all/${sequence_tag}/$(basename "${output_root}")"
      if [[ -f "${source_root}/meta.json" ]]; then
        echo "[copy] ${model} :: ${group_dir} :: davis_all -> ${benchmark} :: ${sequence_tag} :: ${variant}"
        copy_variant_dir "${source_root}" "${output_root}"
        continue
      fi
    fi
    echo "[run] ${model} :: ${group_dir} :: ${benchmark} :: ${sequence_tag} :: ${variant}"
    ran_local=1
    case "${variant}" in
      sam2_prompt)
        run_sam2_sequence "${direct_root}" "${output_root}" "${variant_name}"
        ;;
      upsample_anything)
        run_upa_sequence "${direct_root}" "${output_root}" "${variant_name}"
        ;;
      *)
        echo "[error] unsupported variant: ${variant}" >&2
        exit 1
        ;;
    esac
  done

  return "${ran_local}"
}

main() {
  IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
  IFS=',' read -r -a model_array <<< "${MODELS}"
  local ran_any=0

  for raw_model in "${model_array[@]}"; do
    model="$(canonical_model_name "$(trim "${raw_model}")")"
    [[ -n "${model}" ]] || continue
    group_dir="$(group_dir_for_model "${model}")"
    while IFS= read -r benchmark; do
      [[ -n "${benchmark}" ]] || continue
      while IFS= read -r sequence_root; do
        [[ -n "${sequence_root}" ]] || continue
        sequence_tag="$(basename "${sequence_root}")"
        if run_refined_sequence_variants "${model}" "${group_dir}" "${benchmark}" "${sequence_root}" "${sequence_tag}"; then
          ran_any=1
        fi
      done < <(sequence_roots_for_scope "${model}" "${benchmark}")
    done < <(emit_ordered_benchmarks)
  done

  if [[ "${ran_any}" == "0" ]]; then
    echo "[warn] no refined sequences were scheduled. Expected bilinear masks under: ${MASK_ROOT}/<model>/<group>/<benchmark>/<sequence>/bilinear_upsample"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
