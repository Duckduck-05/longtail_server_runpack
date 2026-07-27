#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CM_REPO="${LTX_REPOS_ROOT:-$ROOT/third_party}/ImbDiff-CM"
WEIGHTS="$CM_REPO/stats/pt_inception-2015-12-05-6726825d.pth"
METADATA="$WEIGHTS.ltx.json"
mkdir -p "$(dirname "$WEIGHTS")"
if [[ ! -s "$WEIGHTS" ]]; then
  curl --fail --location --retry 3 --output "$WEIGHTS" \
    "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
fi
python - "$WEIGHTS" "$METADATA" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
if p.stat().st_size < 10_000_000:
    raise SystemExit(f"invalid FID weight download: {p} ({p.stat().st_size} bytes)")
sha = hashlib.sha256(p.read_bytes()).hexdigest()
meta = Path(sys.argv[2])
meta.write_text(json.dumps({
    "asset": p.name,
    "source": "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
    "bytes": p.stat().st_size,
    "sha256": sha,
}, indent=2) + "\n", encoding="utf-8")
print(f"CM FID Inception ready: {p} ({p.stat().st_size} bytes, sha256={sha})")
PY
