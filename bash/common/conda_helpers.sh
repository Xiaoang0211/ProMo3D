#!/usr/bin/env bash

conda_python_default() {
  local explicit_flag="${1:-0}"
  if [[ "${explicit_flag}" != "1" ]]; then
    export PYTHON_BIN="python"
  fi
}

ensure_conda_loaded() {
  if [[ "$(type -t conda 2>/dev/null || true)" == "function" ]]; then
    return 0
  fi
  local conda_sh="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
  if [[ -f "${conda_sh}" ]]; then
    # shellcheck source=/dev/null
    source "${conda_sh}"
  fi
  if [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
    echo "[error] conda shell function not available; set CONDA_SH to your conda.sh path." >&2
    return 1
  fi
}

activate_conda_env() {
  local env_name="${1:-}"
  local explicit_python_flag="${2:-0}"
  [[ "${explicit_python_flag}" == "1" ]] && return 0
  [[ -n "${env_name}" ]] || return 0
  ensure_conda_loaded
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${env_name}" ]]; then
    conda activate "${env_name}"
  fi
  conda_python_default "${explicit_python_flag}"
}

model_conda_env_name() {
  local model="${1:?model required}"
  case "${model}" in
    pi3) echo "${PI3_CONDA_ENV:-pi3}" ;;
    pi3x) echo "${PI3X_CONDA_ENV:-pi3}" ;;
    vggt) echo "${VGGT_CONDA_ENV:-vggt}" ;;
    vggt_omega|vggtomega|omega) echo "${VGGT_OMEGA_CONDA_ENV:-vggt}" ;;
    depth_anything_3|da3) echo "${DA3_CONDA_ENV:-depthanything3}" ;;
    map_anything|mapanything) echo "${MAPANYTHING_CONDA_ENV:-mapanything}" ;;
    dust3r) echo "${DUST3R_CONDA_ENV:-dust3r}" ;;
    *) echo "" ;;
  esac
}

canonical_model_name() {
  local model="${1:?model required}"
  case "${model}" in
    da3) echo "depth_anything_3" ;;
    mapanything) echo "map_anything" ;;
    vggtomega|omega) echo "vggt_omega" ;;
    *) echo "${model}" ;;
  esac
}
