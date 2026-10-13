#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEOMOTION_ROOT="${GEOMOTION_ROOT:-/home/xiaoang/FF_Reconstruction/GeoMotion}"
RUNTIME_ROOT="${GEOMOTION_RUNTIME_ROOT:-/media/xiaoang/f66ba4e1-c36b-4386-9589-1efb0a906864/third_party_runtime/geomotion}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaoang/anaconda3/envs/vggt4d/bin/python}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p}"
MODEL_PATH="${GEOMOTION_MODEL_PATH:-${RUNTIME_ROOT}/hf/best_model.pth}"
PI3_MODEL_PATH="${PI3_MODEL_PATH:-${ROOT_DIR}/models/Pi3/ckpt/model_pi3.safetensors}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${RUNTIME_ROOT}/cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-${ROOT_DIR}/sam2/checkpoints/sam2.1_hiera_large.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/experiments/runtime/geomotion}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-geomotion-runtime}"
export PYTHONPATH="${GEOMOTION_ROOT}:${GEOMOTION_ROOT}/sam2-main${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m isdynamic.viz.profile_geomotion_runtime \
  --geomotion_root "${GEOMOTION_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --pi3_model_path "${PI3_MODEL_PATH}" \
  --raft_model_path "${RAFT_MODEL_PATH}" \
  --sam2_checkpoint "${SAM2_CHECKPOINT}" \
  --davis_root "${DAVIS_ROOT}" \
  --sequence "${SEQUENCE:-parkour}" \
  --warmup_sequence "${WARMUP_SEQUENCE:-cows}" \
  --num_frames "${NUM_FRAMES:-100}" \
  --warmup_frames "${WARMUP_FRAMES:-10}" \
  --sequence_length "${SEQUENCE_LENGTH:-16}" \
  --output_dir "${OUTPUT_DIR}"
