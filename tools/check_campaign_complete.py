#!/usr/bin/env python3
"""Check that every row in a campaign has a successful published FID."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx.completion import check_campaign_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    campaign, incomplete = check_campaign_complete(args.config)
    if incomplete:
        print(f"[main-table] {campaign.raw['campaign']['name']}: {len(incomplete)}/{len(campaign.tasks)} incomplete", file=sys.stderr)
        for item in incomplete[:20]:
            print(f"[main-table] {item}", file=sys.stderr)
        if len(incomplete) > 20:
            print(f"[main-table] ... {len(incomplete) - 20} more", file=sys.stderr)
        return 1
    print(f"[main-table] {campaign.raw['campaign']['name']}: {len(campaign.tasks)} rows complete with SUCCESS + generation/FID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
