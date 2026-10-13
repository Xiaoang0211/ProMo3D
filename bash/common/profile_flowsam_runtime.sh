#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaoang/anaconda3/envs/pi3/bin/python}"
FLOWSAM_ROOT="${FLOWSAM_ROOT:-/home/xiaoang/FF_Reconstruction/flowsam}"
RUNTIME_ROOT="${FLOWSAM_RUNTIME_ROOT:-/media/xiaoang/f66ba4e1-c36b-4386-9589-1efb0a906864/third_party_runtime/flowsam}"
CHECKPOINT_ROOT="${FLOWSAM_CHECKPOINT_ROOT:-${RUNTIME_ROOT}/checkpoints}"
SEGMENT_ANYTHING_ROOT="${SEGMENT_ANYTHING_ROOT:-${RUNTIME_ROOT}/segment-anything}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/experiments/runtime/flowsam}"
SEQUENCE="${SEQUENCE:-parkour}"
WARMUP_SEQUENCE="${WARMUP_SEQUENCE:-cows}"
NUM_FRAMES="${NUM_FRAMES:-100}"
WARMUP_FRAMES="${WARMUP_FRAMES:-10}"
FLOW_GAPS="${FLOW_GAPS:-1,-1,2,-2}"
RUN_STAGES="${RUN_STAGES:-raft,flowpsam,flowisam,summary}"

RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-${FLOWSAM_ROOT}/flow/raft-things.pth}"
FLOWPSAM_CHECKPOINT="${FLOWPSAM_CHECKPOINT:-${CHECKPOINT_ROOT}/frame_level_flowpsam_vitbvith_train_on_oclrsyn_dvs17.pth}"
FLOWISAM_CHECKPOINT="${FLOWISAM_CHECKPOINT:-${CHECKPOINT_ROOT}/frame_level_flowisam_vith_train_on_oclrsyn_dvs17m.pth}"
SAM_VIT_H_CHECKPOINT="${SAM_VIT_H_CHECKPOINT:-${CHECKPOINT_ROOT}/sam_vit_h_4b8939.pth}"
SAM_VIT_B_CHECKPOINT="${SAM_VIT_B_CHECKPOINT:-${CHECKPOINT_ROOT}/sam_vit_b_01ec64.pth}"

for required_path in \
  "${RAFT_CHECKPOINT}" \
  "${FLOWPSAM_CHECKPOINT}" \
  "${FLOWISAM_CHECKPOINT}" \
  "${SAM_VIT_H_CHECKPOINT}" \
  "${SAM_VIT_B_CHECKPOINT}" \
  "${SEGMENT_ANYTHING_ROOT}/segment_anything/build_sam.py"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required Flow-SAM runtime asset: ${required_path}" >&2
    exit 1
  fi
done

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a stages <<< "${RUN_STAGES}"
for stage in "${stages[@]}"; do
  case "${stage}" in
    raft)
      "${PYTHON_BIN}" -m isdynamic.viz.profile_flowsam_raft_runtime \
        --flowsam_root "${FLOWSAM_ROOT}" \
        --raft_checkpoint "${RAFT_CHECKPOINT}" \
        --davis_root "${DAVIS_ROOT}" \
        --sequence "${SEQUENCE}" \
        --warmup_sequence "${WARMUP_SEQUENCE}" \
        --num_frames "${NUM_FRAMES}" \
        --warmup_frames "${WARMUP_FRAMES}" \
        --flow_gaps "${FLOW_GAPS}" \
        --output_dir "${OUTPUT_DIR}"
      ;;
    flowpsam)
      "${PYTHON_BIN}" -m isdynamic.viz.profile_flowsam_decoder_runtime \
        --model flowpsam \
        --flowsam_root "${FLOWSAM_ROOT}" \
        --segment_anything_root "${SEGMENT_ANYTHING_ROOT}" \
        --flowsam_checkpoint "${FLOWPSAM_CHECKPOINT}" \
        --sam_vit_h_checkpoint "${SAM_VIT_H_CHECKPOINT}" \
        --sam_vit_b_checkpoint "${SAM_VIT_B_CHECKPOINT}" \
        --davis_root "${DAVIS_ROOT}" \
        --flow_cache_root "${OUTPUT_DIR}/flow_cache" \
        --sequence "${SEQUENCE}" \
        --warmup_sequence "${WARMUP_SEQUENCE}" \
        --num_frames "${NUM_FRAMES}" \
        --warmup_frames "${WARMUP_FRAMES}" \
        --flow_gaps "${FLOW_GAPS}" \
        --output_dir "${OUTPUT_DIR}"
      ;;
    flowisam)
      "${PYTHON_BIN}" -m isdynamic.viz.profile_flowsam_decoder_runtime \
        --model flowisam \
        --flowsam_root "${FLOWSAM_ROOT}" \
        --segment_anything_root "${SEGMENT_ANYTHING_ROOT}" \
        --flowsam_checkpoint "${FLOWISAM_CHECKPOINT}" \
        --sam_vit_h_checkpoint "${SAM_VIT_H_CHECKPOINT}" \
        --sam_vit_b_checkpoint "${SAM_VIT_B_CHECKPOINT}" \
        --davis_root "${DAVIS_ROOT}" \
        --flow_cache_root "${OUTPUT_DIR}/flow_cache" \
        --sequence "${SEQUENCE}" \
        --warmup_sequence "${WARMUP_SEQUENCE}" \
        --num_frames "${NUM_FRAMES}" \
        --warmup_frames "${WARMUP_FRAMES}" \
        --flow_gaps "${FLOW_GAPS}" \
        --output_dir "${OUTPUT_DIR}"
      ;;
    summary)
      "${PYTHON_BIN}" -m isdynamic.viz.summarize_flowsam_runtime \
        --raft_csv "${OUTPUT_DIR}/raft_per_frame.csv" \
        --raft_summary "${OUTPUT_DIR}/raft_summary.json" \
        --flowpsam_csv "${OUTPUT_DIR}/flowpsam_per_frame.csv" \
        --flowpsam_summary "${OUTPUT_DIR}/flowpsam_summary.json" \
        --flowisam_csv "${OUTPUT_DIR}/flowisam_per_frame.csv" \
        --flowisam_summary "${OUTPUT_DIR}/flowisam_summary.json" \
        --output_dir "${OUTPUT_DIR}"
      ;;
    *)
      echo "Unknown stage: ${stage}. Use raft,flowpsam,flowisam,summary." >&2
      exit 1
      ;;
  esac
done
