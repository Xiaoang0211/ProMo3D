#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEGANYMO_ROOT="${SEGANYMO_ROOT:-/home/xiaoang/FF_Reconstruction/SegAnyMo}"
RUNTIME_ROOT="${SEGANYMO_RUNTIME_ROOT:-/media/xiaoang/f66ba4e1-c36b-4386-9589-1efb0a906864/third_party_runtime/seganymo}"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaoang/anaconda3/envs/vggt4d/bin/python}"
DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p}"
MOTION_CHECKPOINT="${SEGANYMO_MOTION_CHECKPOINT:-${RUNTIME_ROOT}/checkpoints/moseg.pth}"
TAPIR_CHECKPOINT="${SEGANYMO_TAPIR_CHECKPOINT:-${SEGANYMO_ROOT}/preproc/checkpoints/bootstapir_checkpoint_v2.pt}"
SAM2_CHECKPOINT="${SEGANYMO_SAM2_CHECKPOINT:-${SEGANYMO_ROOT}/sam2/checkpoints/sam2_hiera_large.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/experiments/runtime/seganymo}"
PYTHON_PACKAGES="${SEGANYMO_PYTHON_PACKAGES:-${RUNTIME_ROOT}/python_packages}"
CACHE_ROOT="${SEGANYMO_CACHE_ROOT:-${RUNTIME_ROOT}/cache}"

export PYTHONPATH="${PYTHON_PACKAGES}:${SEGANYMO_ROOT}:${SEGANYMO_ROOT}/preproc:${SEGANYMO_ROOT}/sam2${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-seganymo-runtime}"

mkdir -p "${OUTPUT_DIR}"
if [[ -s "${OUTPUT_DIR}/summary.json" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "SegAnyMo runtime summary already exists: ${OUTPUT_DIR}/summary.json"
  exit 0
fi

exec 9>"${OUTPUT_DIR}/.runtime.lock"
if ! flock -n 9; then
  echo "SegAnyMo runtime profiler is already running for ${OUTPUT_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m isdynamic.viz.profile_seganymo_runtime \
  --seganymo_root "${SEGANYMO_ROOT}" \
  --motion_checkpoint "${MOTION_CHECKPOINT}" \
  --tapir_checkpoint "${TAPIR_CHECKPOINT}" \
  --sam2_checkpoint "${SAM2_CHECKPOINT}" \
  --davis_root "${DAVIS_ROOT}" \
  --sequence "${SEQUENCE:-parkour}" \
  --warmup_sequence "${WARMUP_SEQUENCE:-cows}" \
  --num_frames "${NUM_FRAMES:-100}" \
  --warmup_frames "${WARMUP_FRAMES:-10}" \
  --step "${STEP:-8}" \
  --grid_size "${GRID_SIZE:-10}" \
  --seed "${SEED:-0}" \
  --output_dir "${OUTPUT_DIR}"
