#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASETS_DIR="${DATASETS_DIR:-${ROOT_DIR}/datasets}"
DOWNLOADS_DIR="${DOWNLOADS_DIR:-${DATASETS_DIR}/downloads}"

DOWNLOAD_PMD_160="${DOWNLOAD_PMD_160:-1}"
DOWNLOAD_PMD_10="${DOWNLOAD_PMD_10:-0}"
DOWNLOAD_DAVIS="${DOWNLOAD_DAVIS:-1}"
DOWNLOAD_SEGTRACK="${DOWNLOAD_SEGTRACK:-1}"
DOWNLOAD_FBMS="${DOWNLOAD_FBMS:-1}"

# PMD subset assets
PMD_10_URL="${PMD_10_URL:-https://drive.google.com/file/d/1J4Rz5tzXb_qIZ2t0-0TaQlIuv8zFWBGN/view?usp=drive_link}"

# Evaluation datasets
DAVIS_VARIANT="${DAVIS_VARIANT:-semisupervised}"   # semisupervised | unsupervised
DAVIS_UNSUP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip"
DAVIS_SEMISUP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
SEGTRACK_URL="${SEGTRACK_URL:-https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip}"

FBMS_PAGE_URL="${FBMS_PAGE_URL:-https://lmb.informatik.uni-freiburg.de/resources/datasets}"
FBMS_TRAIN_URL="${FBMS_TRAIN_URL:-https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip}"
FBMS_TEST_URL="${FBMS_TEST_URL:-https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_http() {
  local url="$1"
  local dst="$2"
  if have_cmd curl; then
    curl -L --fail "$url" -o "$dst"
    return
  fi
  if have_cmd wget; then
    wget -O "$dst" "$url"
    return
  fi
  echo "[error] need curl or wget to download: $url" >&2
  exit 1
}

download_gdrive() {
  local url="$1"
  local dst="$2"
  if ! have_cmd gdown; then
    echo "[error] gdown is required for Google Drive downloads." >&2
    echo "[hint] install it with: pip install gdown" >&2
    exit 1
  fi
  gdown --fuzzy "$url" -O "$dst"
}

ensure_download() {
  local kind="$1"
  local url="$2"
  local dst="$3"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    echo "[skip] archive already exists: $dst"
    return
  fi
  echo "[download] $kind -> $dst"
  if [[ "$url" == *"drive.google.com"* ]]; then
    download_gdrive "$url" "$dst"
  else
    download_http "$url" "$dst"
  fi
}

extract_zip_if_needed() {
  local archive="$1"
  local outdir="$2"
  local marker="$3"
  mkdir -p "$outdir"
  if [[ -e "$marker" ]]; then
    echo "[skip] already extracted: $marker"
    return
  fi
  if ! have_cmd unzip; then
    echo "[error] unzip is required to extract: $archive" >&2
    exit 1
  fi
  echo "[extract] $archive -> $outdir"
  unzip -q -n "$archive" -d "$outdir"
}

normalize_fbms_layout() {
  local fbms_root="$1"
  local train_dir="${fbms_root}/Trainingset"
  local test_dir="${fbms_root}/Testset"

  if [[ -d "${train_dir}" && -d "${test_dir}" ]]; then
    return
  fi

  if [[ -d "${fbms_root}/FBMS_Trainingset" && ! -d "${train_dir}" ]]; then
    mv "${fbms_root}/FBMS_Trainingset" "${train_dir}"
  fi
  if [[ -d "${fbms_root}/FBMS_Testset" && ! -d "${test_dir}" ]]; then
    mv "${fbms_root}/FBMS_Testset" "${test_dir}"
  fi

  if [[ ! -d "${train_dir}" ]]; then
    nested_train="$(find "${fbms_root}" -maxdepth 3 -type d \( -name 'Trainingset' -o -name 'FBMS_Trainingset' \) | head -n 1 || true)"
    if [[ -n "${nested_train}" && "${nested_train}" != "${train_dir}" ]]; then
      mv "${nested_train}" "${train_dir}"
    fi
  fi
  if [[ ! -d "${test_dir}" ]]; then
    nested_test="$(find "${fbms_root}" -maxdepth 3 -type d \( -name 'Testset' -o -name 'FBMS_Testset' \) | head -n 1 || true)"
    if [[ -n "${nested_test}" && "${nested_test}" != "${test_dir}" ]]; then
      mv "${nested_test}" "${test_dir}"
    fi
  fi

  if [[ ! -d "${train_dir}" || ! -d "${test_dir}" ]]; then
    echo "[error] FBMS extraction did not produce the expected layout under ${fbms_root}" >&2
    echo "[error] Expected directories: Trainingset/ and Testset/" >&2
    echo "[error] This is the layout required by PMD." >&2
    exit 1
  fi
}

main() {
  mkdir -p "$DATASETS_DIR" "$DOWNLOADS_DIR"

  if [[ "$DOWNLOAD_PMD_160" == "1" ]]; then
    if [[ -d "${DATASETS_DIR}/youtubevos_160vids/train/Annotations" && -d "${DATASETS_DIR}/youtubevos_160vids/validation/Annotations" ]]; then
      echo "[skip] PMD source subset already exists: ${DATASETS_DIR}/youtubevos_160vids"
    else
      bash "${ROOT_DIR}/scripts/download_huggingface.sh" --assets data --data-root "${DATASETS_DIR}/youtubevos_160vids"
    fi
  fi

  if [[ "$DOWNLOAD_PMD_10" == "1" ]]; then
    local_archive="${DOWNLOADS_DIR}/pmd_youtubevos_10_train.zip"
    ensure_download "PMD 10-train subset" "$PMD_10_URL" "$local_archive"
    extract_zip_if_needed "$local_archive" "$DATASETS_DIR" "${DATASETS_DIR}/youtubevos_10vids"
  fi

  if [[ "$DOWNLOAD_DAVIS" == "1" ]]; then
    case "$DAVIS_VARIANT" in
      unsupervised)
        davis_url="$DAVIS_UNSUP_URL"
        davis_archive="${DOWNLOADS_DIR}/DAVIS-2017-Unsupervised-trainval-480p.zip"
        ;;
      semisupervised)
        davis_url="$DAVIS_SEMISUP_URL"
        davis_archive="${DOWNLOADS_DIR}/DAVIS-2017-trainval-480p.zip"
        ;;
      *)
        echo "[error] unsupported DAVIS_VARIANT=$DAVIS_VARIANT (expected: unsupervised or semisupervised)" >&2
        exit 1
        ;;
    esac
    ensure_download "DAVIS 2017 ${DAVIS_VARIANT} 480p" "$davis_url" "$davis_archive"
    extract_zip_if_needed "$davis_archive" "$DATASETS_DIR" "${DATASETS_DIR}/DAVIS"
  fi

  if [[ "$DOWNLOAD_SEGTRACK" == "1" ]]; then
    segtrack_archive="${DOWNLOADS_DIR}/SegTrackv2.zip"
    ensure_download "SegTrackv2" "$SEGTRACK_URL" "$segtrack_archive"
    extract_zip_if_needed "$segtrack_archive" "$DATASETS_DIR" "${DATASETS_DIR}/SegTrackv2"
  fi

  if [[ "$DOWNLOAD_FBMS" == "1" ]]; then
    fbms_root="${DATASETS_DIR}/FBMS-59"
    mkdir -p "$fbms_root"
    fbms_train_archive="${DOWNLOADS_DIR}/FBMS-59-train.zip"
    fbms_test_archive="${DOWNLOADS_DIR}/FBMS-59-test.zip"
    ensure_download "FBMS-59 training set" "$FBMS_TRAIN_URL" "$fbms_train_archive"
    ensure_download "FBMS-59 test set" "$FBMS_TEST_URL" "$fbms_test_archive"
    extract_zip_if_needed "$fbms_train_archive" "$fbms_root" "${fbms_root}/Trainingset"
    extract_zip_if_needed "$fbms_test_archive" "$fbms_root" "${fbms_root}/Testset"
    normalize_fbms_layout "$fbms_root"
  fi

  echo "[done] dataset download/extraction step finished"
  echo "[info] datasets root: $DATASETS_DIR"
}

main "$@"
