from pathlib import Path

from .base import Adapter
from .ccua import CCUAAdapter
from .coral import CoralAdapter
from .oc import OCAdapter
from .cm import CMAdapter
from .t2h_unified import T2HUnifiedAdapter


def make_adapter(name: str, root: Path) -> Adapter:
    mapping = {
        "coral": CoralAdapter,
        "oc": OCAdapter,
        "cm": CMAdapter,
        "ccua": CCUAAdapter,
        "t2h_unified": T2HUnifiedAdapter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown adapter: {name}")
    return mapping[name](root)
