#!/usr/bin/env bash
# Fetch the source papers for the six methods in the unified CIFAR-LT table.
#
# PDFs are not committed (see papers/.gitignore): they are third-party
# copyrighted works and one is ~23 MB. This script re-fetches them on demand
# so the runpack stays self-describing without redistributing the files.
#
# Re-running is safe: an existing, non-empty PDF is left alone.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

fetch() {
  local out="$1" url="$2" label="$3"
  if [[ -s "$out" ]]; then
    echo "[papers] have    $out"
    return 0
  fi
  echo "[papers] fetching $out  <- $label"
  if curl -fsSL --retry 3 --retry-delay 2 -A "longtail-runpack/1.0" -o "$out.part" "$url"; then
    # Guard against a captive portal or error page saved as a .pdf.
    if [[ "$(head -c 4 "$out.part")" == "%PDF" ]]; then
      mv "$out.part" "$out"
    else
      rm -f "$out.part"
      echo "[papers] WARN    $out: response was not a PDF; skipped" >&2
      return 1
    fi
  else
    rm -f "$out.part"
    echo "[papers] WARN    $out: download failed; skipped" >&2
    return 1
  fi
}

# OpenReview serves its PDFs behind a bot challenge, so T2H and CM cannot be
# fetched non-interactively. Neither is on arXiv and Semantic Scholar lists no
# open-access mirror. Copy from a local path if one is configured, otherwise
# print the URL for a manual (browser) download.
local_or_manual() {
  local out="$1" src="$2" url="$3"
  if [[ -s "$out" ]]; then
    echo "[papers] have    $out"
    return 0
  fi
  if [[ -n "$src" && -s "$src" ]]; then
    cp "$src" "$out"
    echo "[papers] copied  $out  <- $src"
    return 0
  fi
  echo "[papers] MANUAL  $out: OpenReview blocks scripted downloads." >&2
  echo "[papers]         open $url in a browser and save it as papers/$out" >&2
  return 1
}

status=0

# DDPM — the unconditional/conditional backbone every row builds on.
fetch ddpm-neurips2020-ho.pdf \
  "https://arxiv.org/pdf/2006.11239" "arXiv:2006.11239" || status=1

# CBDM — CVPR 2023. Also the codebase T2H and IGD-ML are derived from.
fetch cbdm-cvpr2023-qin.pdf \
  "https://arxiv.org/pdf/2305.00562" "arXiv:2305.00562" || status=1

# T2H — ICLR 2024 poster. Not on arXiv; OpenReview is the source of record.
local_or_manual t2h-iclr2024-zhang.pdf \
  "${LTX_PAPER_T2H_SRC:-}" "https://openreview.net/forum?id=NW2s5XXwXU" || status=1

# CM (ImbDiff-CM) — ICLR 2026 Oral. Not on arXiv.
local_or_manual cm-iclr2026-hong.pdf \
  "${LTX_PAPER_CM_SRC:-}" "https://openreview.net/forum?id=wSGle6ag5I" || status=1

# CORAL — NeurIPS 2025. A copy is also vendored at
# third_party/coral-lt-diffusion/CORAL-NeurIPS2025-Rodriguezetal.pdf.
fetch coral-neurips2025-rodriguez.pdf \
  "https://arxiv.org/pdf/2506.15933" "arXiv:2506.15933" || status=1

# CCUA — arXiv preprint (v3, Jun 2026). The repository ships a U-Net (DDPM) and
# a DiT/SiT pipeline; only the U-Net one is in this table.
fetch ccua-arxiv2507.09052-chen.pdf \
  "https://arxiv.org/pdf/2507.09052" "arXiv:2507.09052" || status=1

echo
if [[ "$status" -eq 0 ]]; then
  echo "[papers] all 6 papers present."
else
  echo "[papers] some downloads failed; see warnings above." >&2
fi
exit "$status"
