from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import LoadedCampaign, load_campaign


def check_campaign_complete(config_path: str | Path) -> Tuple[LoadedCampaign, List[str]]:
    """Return the campaign and a concise list of incomplete metric rows.

    A successful process alone is not enough for the main table gate: a row
    must also have the collected FID that the report would publish.
    """
    campaign = load_campaign(config_path)
    incomplete: List[str] = []
    for task in campaign.tasks:
        run_dir = Path(task.run_dir)
        label = f"{task.dataset.get('name', task.stage)}/{task.method}/seed{task.seed}"
        if not (run_dir / "SUCCESS").is_file():
            incomplete.append(f"{label}: SUCCESS missing")
            continue
        metrics_path = run_dir / "metrics.collected.json"
        if not metrics_path.is_file():
            incomplete.append(f"{label}: metrics.collected.json missing")
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            fid = payload.get("generation/FID")
            if fid is None or not np.isfinite(float(fid)):
                incomplete.append(f"{label}: generation/FID missing")
        except Exception as exc:
            incomplete.append(f"{label}: invalid metrics ({exc})")
    return campaign, incomplete
