from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..config import Task


@dataclass
class Phase:
    name: str
    command: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    skip_if_exists: List[Path] = field(default_factory=list)


class Adapter:
    name = "base"

    def __init__(self, root: Path):
        self.root = root

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        raise NotImplementedError

    def repo_dir(self, task: Task) -> Path:
        directory = task.repository.get("directory", self.name)
        path = Path(directory)
        return path if path.is_absolute() else Path(task.runtime["repos_root"]) / path
