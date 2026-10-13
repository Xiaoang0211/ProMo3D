#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaoang/anaconda3/envs/pi3/bin/python}"
PI3X_CHECKPOINT="${PI3X_CHECKPOINT:-${ROOT_DIR}/models/Pi3/ckpt/model_pi3x.safetensors}"
PI3X_PROBE_ROOT="${PI3X_PROBE_ROOT:-/media/xiaoang/637ae275-d5e6-45cb-bf4e-860a889000ab/isDynamic/outputs/model_outputs/pi3/rank_8_pi3x/isdynamic_blockwise_youtubevos_pi3x/pair_lowrank_quadratic}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/experiments/runtime/pi3x}"
SEQUENCE="${SEQUENCE:-parkour}"
WARMUP_SEQUENCE="${WARMUP_SEQUENCE:-cows}"
NUM_FRAMES="${NUM_FRAMES:-100}"
WARMUP_FRAMES="${WARMUP_FRAMES:-10}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
RUN_STAGES="${RUN_STAGES:-coarse,sam2,summary}"
CACHE_IMAGE_FEATURES="${CACHE_IMAGE_FEATURES:-1}"

SAM2_REPO_ROOT="${SAM2_REPO_ROOT:-${ROOT_DIR}/sam2}"
SAM2_CFG="${SAM2_CFG:-sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${SAM2_REPO_ROOT}/checkpoints/sam2.1_hiera_large.pt}"

probe_count="$(
  find "${PI3X_PROBE_ROOT}/inter_frame_block" -mindepth 2 -maxdepth 2 -type f -name best_probe.pt | wc -l
)"
if [[ "${probe_count}" -ne 18 ]]; then
  echo "Expected 18 Pi3X inter-frame rank-8 probe checkpoints, found ${probe_count}: ${PI3X_PROBE_ROOT}" >&2
  exit 1
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-pmd-runtime}"
mkdir -p "${MPLCONFIGDIR}"

IFS=',' read -r -a stages <<< "${RUN_STAGES}"
for stage in "${stages[@]}"; do
  case "${stage}" in
    coarse)
      "${PYTHON_BIN}" -m isdynamic.viz.profile_pi3x_runtime \
        --checkpoint "${PI3X_CHECKPOINT}" \
        --probe_root "${PI3X_PROBE_ROOT}" \
        --davis_root "${DAVIS_ROOT}" \
        --sequence "${SEQUENCE}" \
        --warmup_sequence "${WARMUP_SEQUENCE}" \
        --num_frames "${NUM_FRAMES}" \
        --warmup_frames "${WARMUP_FRAMES}" \
        --context_size "${CONTEXT_SIZE}" \
        --context_stride "${CONTEXT_STRIDE}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_masks
      ;;
    sam2)
      sam2_args=(
        "${PYTHON_BIN}" -m isdynamic.viz.refine_motion_masks_with_sam2
        --mask_root "${OUTPUT_DIR}/coarse_masks/${SEQUENCE}" \
        --runtime_warmup_mask_root "${OUTPUT_DIR}/warmup_masks/${WARMUP_SEQUENCE}" \
        --video_path "${DAVIS_ROOT}/DAVIS/JPEGImages/480p/${SEQUENCE}" \
        --sam2_repo_root "${SAM2_REPO_ROOT}" \
        --sam2_cfg "${SAM2_CFG}" \
        --sam2_checkpoint "${SAM2_CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}/sam2/${SEQUENCE}" \
        --variant_name runtime_sam2_prompt \
        --prompt_score_source upsampled \
        --prompt_threshold 0.9 \
        --prompt_threshold_mode otsu \
        --prompt_threshold_offset 0.2 \
        --prompt_erode_iterations 2 \
        --instance_mean_score_threshold 0.0 \
        --instance_mean_score_threshold_mode otsu \
        --instance_mean_score_threshold_offset 0.3 \
        --propagate_mode parity \
        --window_size 30 \
        --window_overlap 0 \
        --device cuda
      )
      if [[ "${CACHE_IMAGE_FEATURES}" == "1" ]]; then
        sam2_args+=(--cache_image_features)
      fi
      "${sam2_args[@]}"
      ;;
    summary)
      "${PYTHON_BIN}" -m isdynamic.viz.summarize_pi3x_runtime \
        --coarse_csv "${OUTPUT_DIR}/coarse_per_frame.csv" \
        --coarse_summary "${OUTPUT_DIR}/coarse_summary.json" \
        --sam2_meta "${OUTPUT_DIR}/sam2/${SEQUENCE}/meta.json" \
        --output_dir "${OUTPUT_DIR}"
      ;;
    *)
      echo "Unknown stage: ${stage}. Use coarse,sam2,summary." >&2
      exit 1
      ;;
  esac
done
