from pathlib import Path

from .base import Adapter
from .ccua import CCUAAdapter
from .coral import CoralAdapter
from .oc import OCAdapter
from .cm import CMAdapter


def make_adapter(name: str, root: Path) -> Adapter:
    mapping = {"coral": CoralAdapter, "oc": OCAdapter, "cm": CMAdapter, "ccua": CCUAAdapter}
    if name not in mapping:
        raise ValueError(f"Unknown adapter: {name}")
    return mapping[name](root)
