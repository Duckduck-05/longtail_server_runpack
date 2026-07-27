#!/usr/bin/env bash
# Source this file after loading .env.local.  It makes the licensed
# ImageNet-LT payload reproducibly available on a fresh receiving server.
#
# Default path: download the original ILSVRC2012 training archive from the
# official ImageNet endpoint, expand its per-synset tars, then obtain the
# public ImageNet-LT train/validation manifests.  This is the layout used by
# common ImageNet-LT repositories.  A custom, checksum-pinned archive or a
# pre-mounted dataset remains available as an override.
set -euo pipefail

LTX_RUNPACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LTX_DATA_ROOT="${LTX_DATA_ROOT:-$LTX_RUNPACK_ROOT/data}"
export LTX_DATA_ROOT

export LTX_IMAGENET_ROOT="${LTX_IMAGENET_ROOT:-$LTX_DATA_ROOT/imagenet_lt/images}"
export LTX_IMAGENET_LT_TRAIN_MANIFEST="${LTX_IMAGENET_LT_TRAIN_MANIFEST:-$LTX_DATA_ROOT/imagenet_lt/manifests/ImageNet_LT_train.txt}"
export LTX_IMAGENET_LT_REFERENCE_MANIFEST="${LTX_IMAGENET_LT_REFERENCE_MANIFEST:-$LTX_DATA_ROOT/imagenet_lt/manifests/ImageNet_LT_balanced_val.txt}"

readonly LTX_OFFICIAL_ILSVRC2012_TRAIN_URL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
readonly LTX_OFFICIAL_ILSVRC2012_TRAIN_MD5="1d675b47d978889d74fa0da5fadfb00e"
readonly LTX_IMAGENET_LT_MANIFEST_REVISION="cba11b8b0fb91711eeffd5e45311f321f8a88680"
readonly LTX_IMAGENET_LT_TRAIN_URL_DEFAULT="https://raw.githubusercontent.com/Vanint/SADE-AgnosticLT/${LTX_IMAGENET_LT_MANIFEST_REVISION}/data_txt/ImageNet_LT/ImageNet_LT_train.txt"
readonly LTX_IMAGENET_LT_TRAIN_SHA256_DEFAULT="efdbdad4f050237c310b2f354cf95a8b1d7c8d57a63c4ea4bb6bf2bcb012f37f"
readonly LTX_IMAGENET_LT_REFERENCE_URL_DEFAULT="https://raw.githubusercontent.com/Vanint/SADE-AgnosticLT/${LTX_IMAGENET_LT_MANIFEST_REVISION}/data_txt/ImageNet_LT/ImageNet_LT_val.txt"
readonly LTX_IMAGENET_LT_REFERENCE_SHA256_DEFAULT="9af7ba688acf9532ff5845a8fb14a1e97fdc4c51ac48bc72c08ea9c3f7ff142e"

die() {
  echo "[imagenet-lt] $*" >&2
  return 2
}

require_checksum() {
  local kind="$1" label="$2" value="$3"
  case "$kind" in
    sha256) [[ "$value" =~ ^[A-Fa-f0-9]{64}$ ]] || die "$label requires a 64-character SHA-256." ;;
    md5) [[ "$value" =~ ^[A-Fa-f0-9]{32}$ ]] || die "$label requires a 32-character MD5." ;;
    *) die "unsupported checksum type: $kind" ;;
  esac
}

verify_checksum() {
  local kind="$1" label="$2" path="$3" expected="$4" actual
  case "$kind" in
    sha256) actual="$(sha256sum "$path" | awk '{print tolower($1)}')" ;;
    md5) actual="$(md5sum "$path" | awk '{print tolower($1)}')" ;;
    *) die "unsupported checksum type: $kind" ;;
  esac
  [[ "$actual" == "${expected,,}" ]] || die "$label checksum mismatch for $path; delete only that file and rerun."
}

download_verified() {
  local kind="$1" label="$2" url="$3" expected="$4" target="$5"
  require_checksum "$kind" "$label" "$expected"
  mkdir -p "$(dirname "$target")"
  if [[ ! -f "$target" ]]; then
    # Keep the partial file so a disconnected long transfer resumes on retry.
    curl --fail --location --retry 5 --retry-all-errors --continue-at - --silent --show-error \
      --output "${target}.part" "$url"
    verify_checksum "$kind" "$label" "${target}.part" "$expected"
    mv "${target}.part" "$target"
  fi
  verify_checksum "$kind" "$label" "$target" "$expected"
}

archive_format() {
  local configured="${LTX_IMAGENET_ARCHIVE_FORMAT:-auto}" url="$1" clean
  if [[ "$configured" != "auto" ]]; then
    printf '%s\n' "$configured"
    return
  fi
  clean="${url%%\?*}"
  case "$clean" in
    *.tar.gz|*.tgz) printf '%s\n' "tar.gz" ;;
    *.tar.zst) printf '%s\n' "tar.zst" ;;
    *.tar) printf '%s\n' "tar" ;;
    *) die "cannot infer archive format; set LTX_IMAGENET_ARCHIVE_FORMAT to tar, tar.gz, or tar.zst." ;;
  esac
}

extract_archive() {
  local archive="$1" format="$2" root="$3" strip="${LTX_IMAGENET_ARCHIVE_STRIP_COMPONENTS:-0}"
  [[ "$strip" =~ ^[0-9]+$ ]] || die "LTX_IMAGENET_ARCHIVE_STRIP_COMPONENTS must be a non-negative integer."
  [[ ! -e "$root" ]] || die "image root exists but is not a completed dataset: $root"
  local parent stage
  parent="$(dirname "$root")"
  mkdir -p "$parent"
  stage="$(mktemp -d "$parent/.imagenet-lt-stage.XXXXXX")"

  # Refuse archives that attempt to write outside the staging directory.
  if tar -tf "$archive" | awk 'BEGIN { bad=0 } /^\// || /(^|\/)\.\.(\/|$)/ { bad=1 } END { exit bad }'; then :; else
    rm -rf "$stage"
    die "archive contains unsafe paths."
  fi
  case "$format" in
    tar|tar.gz) tar -xf "$archive" -C "$stage" --strip-components="$strip" ;;
    tar.zst) tar --zstd -xf "$archive" -C "$stage" --strip-components="$strip" ;;
    *) rm -rf "$stage"; die "unsupported LTX_IMAGENET_ARCHIVE_FORMAT=$format" ;;
  esac
  if [[ -z "$(find "$stage" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rm -rf "$stage"
    die "archive extracted no files (check LTX_IMAGENET_ARCHIVE_STRIP_COMPONENTS)."
  fi
  mv "$stage" "$root"
}

extract_official_ilsvrc2012_train() {
  local archive="$1" root="$2" parent stage class_archive class_name
  [[ ! -e "$root" ]] || die "image root exists but is not a completed dataset: $root"
  parent="$(dirname "$root")"
  mkdir -p "$parent"
  stage="$(mktemp -d "$parent/.imagenet-lt-stage.XXXXXX")"
  if tar -tf "$archive" | awk 'BEGIN { bad=0 } /^\// || /(^|\/)\.\.(\/|$)/ { bad=1 } END { exit bad }'; then :; else
    rm -rf "$stage"
    die "official ImageNet archive contains unsafe paths."
  fi
  tar -xf "$archive" -C "$stage"
  shopt -s nullglob
  local class_archives=("$stage"/*.tar)
  shopt -u nullglob
  [[ "${#class_archives[@]}" -eq 1000 ]] || { rm -rf "$stage"; die "official ImageNet archive must contain 1,000 synset tar files; found ${#class_archives[@]}."; }
  mkdir -p "$stage/train"
  for class_archive in "${class_archives[@]}"; do
    class_name="$(basename "${class_archive%.tar}")"
    [[ "$class_name" =~ ^n[0-9]{8}$ ]] || { rm -rf "$stage"; die "unexpected ImageNet synset archive: $class_archive"; }
    mkdir -p "$stage/train/$class_name"
    tar -xf "$class_archive" -C "$stage/train/$class_name"
    rm -f "$class_archive"
  done
  mv "$stage" "$root"
}

source_mode="mounted"
if [[ ! -d "$LTX_IMAGENET_ROOT" ]]; then
  source_mode="${LTX_IMAGENET_SOURCE:-auto}"
  case "$source_mode" in
    auto)
      if [[ -n "${LTX_IMAGENET_ARCHIVE_URL:-}" ]]; then source_mode="custom_archive"; else source_mode="official_ilsvrc2012"; fi
      ;;
    official_ilsvrc2012|custom_archive) ;;
    *) die "LTX_IMAGENET_SOURCE must be auto, official_ilsvrc2012, or custom_archive." ;;
  esac
  if [[ "$source_mode" == "official_ilsvrc2012" ]]; then
    official_url="${LTX_OFFICIAL_ILSVRC2012_TRAIN_URL_OVERRIDE:-$LTX_OFFICIAL_ILSVRC2012_TRAIN_URL}"
    official_md5="${LTX_OFFICIAL_ILSVRC2012_TRAIN_MD5_OVERRIDE:-$LTX_OFFICIAL_ILSVRC2012_TRAIN_MD5}"
    cache="$LTX_DATA_ROOT/.download-cache/ILSVRC2012_img_train.tar"
    download_verified md5 "official ILSVRC2012 training archive" "$official_url" "$official_md5" "$cache"
    extract_official_ilsvrc2012_train "$cache" "$LTX_IMAGENET_ROOT"
  else
    archive_url="${LTX_IMAGENET_ARCHIVE_URL:-}"
    archive_sha="${LTX_IMAGENET_ARCHIVE_SHA256:-}"
    [[ -n "$archive_url" ]] || die "custom_archive requires LTX_IMAGENET_ARCHIVE_URL and LTX_IMAGENET_ARCHIVE_SHA256."
    format="$(archive_format "$archive_url")"
    cache="$LTX_DATA_ROOT/.download-cache/imagenet-lt-images.${format}"
    download_verified sha256 "ImageNet-LT image archive" "$archive_url" "$archive_sha" "$cache"
    extract_archive "$cache" "$format" "$LTX_IMAGENET_ROOT"
  fi
fi

if [[ ! -f "$LTX_IMAGENET_LT_TRAIN_MANIFEST" ]]; then
  train_url="${LTX_IMAGENET_LT_TRAIN_MANIFEST_URL:-$LTX_IMAGENET_LT_TRAIN_URL_DEFAULT}"
  train_sha="${LTX_IMAGENET_LT_TRAIN_MANIFEST_SHA256:-$LTX_IMAGENET_LT_TRAIN_SHA256_DEFAULT}"
  download_verified sha256 "ImageNet-LT train manifest" "$train_url" "$train_sha" "$LTX_IMAGENET_LT_TRAIN_MANIFEST"
fi

if [[ ! -f "$LTX_IMAGENET_LT_REFERENCE_MANIFEST" ]]; then
  reference_url="${LTX_IMAGENET_LT_REFERENCE_MANIFEST_URL:-$LTX_IMAGENET_LT_REFERENCE_URL_DEFAULT}"
  reference_sha="${LTX_IMAGENET_LT_REFERENCE_MANIFEST_SHA256:-$LTX_IMAGENET_LT_REFERENCE_SHA256_DEFAULT}"
  download_verified sha256 "ImageNet-LT balanced reference manifest" "$reference_url" "$reference_sha" "$LTX_IMAGENET_LT_REFERENCE_MANIFEST"
fi

echo "[imagenet-lt] dataset files ready under $LTX_IMAGENET_ROOT (source=$source_mode)"
