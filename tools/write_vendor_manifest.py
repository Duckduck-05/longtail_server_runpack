#!/usr/bin/env python3
"""Record the exact source worktrees copied into third_party/.

Run only while assembling a delivery package.  The delivered package contains
no nested git metadata, so this manifest preserves upstream commits, the fact
that local modifications were intentionally included, and deterministic hashes
of the source entrypoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


COMPONENTS = {
    "cbdm": "CBDM-pytorch",
    "igd": "IGD-ML",
    "cm": "ImbDiff-CM",
    "oc": "OC_LT",
    "coral": "coral-lt-diffusion",
}


def capture(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def digest_files(path: Path) -> dict[str, str]:
    result = {}
    for file in sorted(path.rglob("*.py")):
        if any(part in {".git", "__pycache__"} for part in file.parts):
            continue
        result[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args()
    components = {}
    for key, directory in COMPONENTS.items():
        source = args.source_root / directory
        vendored = args.vendor_root / directory
        if not source.joinpath(".git").is_dir() or not vendored.is_dir():
            raise FileNotFoundError(f"missing source or vendor component: {directory}")
        components[key] = {
            "directory": directory,
            "origin": capture(["git", "remote", "get-url", "origin"], source),
            "commit": capture(["git", "rev-parse", "HEAD"], source),
            "source_worktree_status": capture(["git", "status", "--porcelain"], source).splitlines(),
            "vendored_python_sha256": digest_files(vendored),
        }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "copy_policy": "working trees copied without .git, data, stats, __pycache__, or literal download-cache directories named ...",
        "components": components,
    }
    output = args.vendor_root / "THIRD_PARTY_MANIFEST.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
