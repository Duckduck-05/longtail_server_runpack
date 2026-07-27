from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GPU:
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_pct: float
    temperature_c: float
    power_w: float


def query_gpus() -> List[GPU]:
    fields = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        return []
    gpus: List[GPU] = []
    for row in csv.reader(io.StringIO(proc.stdout)):
        if len(row) < 8:
            continue
        try:
            gpus.append(GPU(
                index=int(row[0].strip()), name=row[1].strip(),
                memory_total_mb=float(row[2]), memory_used_mb=float(row[3]), memory_free_mb=float(row[4]),
                utilization_pct=float(row[5]), temperature_c=float(row[6]), power_w=float(row[7] or 0),
            ))
        except ValueError:
            continue
    return gpus


def gpu_metrics(gpu_id: int) -> Dict[str, float]:
    for gpu in query_gpus():
        if gpu.index == gpu_id:
            return {
                "system/gpu_memory_used_gb": gpu.memory_used_mb / 1024,
                "system/gpu_memory_free_gb": gpu.memory_free_mb / 1024,
                "system/gpu_utilization_pct": gpu.utilization_pct,
                "system/gpu_temperature_c": gpu.temperature_c,
                "system/gpu_power_w": gpu.power_w,
            }
    return {}
