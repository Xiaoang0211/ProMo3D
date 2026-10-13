#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUTS_DIR="${OUTPUTS_DIR:-${ROOT_DIR}/outputs_aspectpad}"
export OUTPUTS_DIR
export ISDYNAMIC_OUTPUTS_ROOT="${ISDYNAMIC_OUTPUTS_ROOT:-${OUTPUTS_DIR}}"

# shellcheck source=/dev/null
source "${ROOT_DIR}/bash/common/conda_helpers.sh"

MODELS="${MODELS:-vggt,vggt_omega,depth_anything_3,mapanything,dust3r}"
BENCHMARKS="${BENCHMARKS:-davis_all,segtrackv2,fbms59}"
OUT_DIR="${OUT_DIR:-${OUTPUTS_DIR}/geometry_padding_stability}"
DEVICE="${DEVICE:-cuda}"
CONTEXT_SIZE="${CONTEXT_SIZE:-3}"
CONTEXT_STRIDE="${CONTEXT_STRIDE:-2}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-5}"
DEPTH_COMPARE_SIZE="${DEPTH_COMPARE_SIZE:-128}"

DAVIS_ROOT="${DAVIS_ROOT:-/home/xiaoang/Datasets/DAVIS-2017-trainval-480p/DAVIS}"
SEGTRACK_ROOT="${SEGTRACK_ROOT:-/home/xiaoang/Datasets/SegTrackv2}"
FBMS_ROOT="${FBMS_ROOT:-/home/xiaoang/Datasets/FBMS-59}"

VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-${ROOT_DIR}/models/vggt/ckpt/model.pt}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-${ROOT_DIR}/models/vggt-omega/checkpoints/vggt_omega_1b_512.pt}"
DA3_MODEL_DIR="${DA3_MODEL_DIR:-${ROOT_DIR}/models/Depth-Anything-3/src/depth_anything_3/ckpt/DA3-giant-1.1}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-518}"
DA3_PROCESS_RES_METHOD="${DA3_PROCESS_RES_METHOD:-upper_bound_resize}"
MAPANYTHING_CHECKPOINT="${MAPANYTHING_CHECKPOINT:-${ROOT_DIR}/models/map-anything/ckpt}"
MAPANYTHING_MACHINE="${MAPANYTHING_MACHINE:-aws}"
MAPANYTHING_RESOLUTION_SET="${MAPANYTHING_RESOLUTION_SET:-518}"
DUST3R_WEIGHTS="${DUST3R_WEIGHTS:-${ROOT_DIR}/models/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
DUST3R_IMAGE_SIZE="${DUST3R_IMAGE_SIZE:-512}"

trim() {
  echo "$1" | xargs
}

run_one_model() {
  local model="$1"
  local python_bin="${PYTHON_BIN:-python3}"
  local python_bin_set_by_user=0
  if [[ -n "${PYTHON_BIN+x}" ]]; then
    python_bin_set_by_user=1
  fi
  activate_conda_env "$(model_conda_env_name "${model}")" "${python_bin_set_by_user}"
  python_bin="${PYTHON_BIN}"

  echo "== padding geometry stability :: ${model} =="
  "${python_bin}" -m isdynamic.viz.evaluate_padding_geometry_stability \
    --models "${model}" \
    --benchmarks "${BENCHMARKS}" \
    --davis_root "${DAVIS_ROOT}" \
    --segtrack_root "${SEGTRACK_ROOT}" \
    --fbms_root "${FBMS_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --context_size "${CONTEXT_SIZE}" \
    --context_stride "${CONTEXT_STRIDE}" \
    --max_sequences "${MAX_SEQUENCES}" \
    --max_frames_per_sequence "${MAX_FRAMES_PER_SEQUENCE}" \
    --depth_compare_size "${DEPTH_COMPARE_SIZE}" \
    --device "${DEVICE}" \
    --vggt_checkpoint "${VGGT_CHECKPOINT}" \
    --vggt_omega_checkpoint "${VGGT_OMEGA_CHECKPOINT}" \
    --da3_model_dir "${DA3_MODEL_DIR}" \
    --da3_process_res "${DA3_PROCESS_RES}" \
    --da3_process_res_method "${DA3_PROCESS_RES_METHOD}" \
    --mapanything_checkpoint "${MAPANYTHING_CHECKPOINT}" \
    --mapanything_machine "${MAPANYTHING_MACHINE}" \
    --mapanything_resolution_set "${MAPANYTHING_RESOLUTION_SET}" \
    --dust3r_weights "${DUST3R_WEIGHTS}" \
    --dust3r_image_size "${DUST3R_IMAGE_SIZE}"
}

IFS=',' read -r -a model_array <<< "${MODELS}"
for raw_model in "${model_array[@]}"; do
  model="$(canonical_model_name "$(trim "${raw_model}")")"
  [[ -n "${model}" ]] || continue
  run_one_model "${model}"
done

python - <<PY
import csv
from pathlib import Path

out_dir = Path("${OUT_DIR}")
for name in ("per_frame", "summary"):
    paths = sorted(out_dir.glob(f"*/{name}.csv"))
    rows = []
    fieldnames = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in seen:
                    seen.add(field)
                    fieldnames.append(field)
            rows.extend(reader)
    if not rows:
        continue
    out_path = out_dir / f"{name}_all_models.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved combined {name}: {out_path}")
PY
