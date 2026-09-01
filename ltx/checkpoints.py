"""Small, framework-free helpers for explicit checkpoint provenance.

Checkpoint discovery is deliberately separate from checkpoint loading.  A run
may discover a checkpoint in its own run directory for the upstream resume
path, but a checkpoint outside that directory is only ever accepted when the
caller names it explicitly in config or on the CLI.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_RESUME_MODE = "full"
RESUME_MODES = ("full", "ema_only")
_MODE_ALIASES = {
    "ema_warm_start": "ema_only",
}
# Accept both the legacy ``ckpt_200000.pt`` and namespaced unified artifacts
# such as ``ckpt_unified_v2_200000.pt``.  Arbitrary renamed files still require
# an explicit resume_step, so a path can never silently pick an unrelated step.
_CHECKPOINT_NAME = re.compile(r"^(?:ckpt_|ckpt_[^/]*_)(\d+)\.pt$")


def normalize_resume_mode(value: Any) -> str:
    """Return the canonical explicit resume mode or raise an actionable error."""
    mode = str(value or DEFAULT_RESUME_MODE).strip().lower()
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in RESUME_MODES:
        choices = ", ".join(RESUME_MODES)
        raise ValueError(f"resume_mode={mode!r} is invalid; choose one of: {choices}")
    return mode


def get_resume_spec(
    train: Mapping[str, Any], method_config: Mapping[str, Any] | None = None
) -> tuple[Path | None, str]:
    """Resolve an explicit checkpoint path and mode from task configuration.

    Method-level values are useful for a multi-method stage and take
    precedence over stage-level values.  A mode without a path is rejected so
    a warm-start setting can never be silently ignored by an adapter.
    """
    values: dict[str, Any] = dict(train)
    if method_config:
        for key in ("resume_checkpoint", "resume_mode", "resume_step"):
            if key in method_config:
                values[key] = method_config[key]

    mode = normalize_resume_mode(values.get("resume_mode", DEFAULT_RESUME_MODE))
    raw_path = str(values.get("resume_checkpoint", "") or "").strip()
    if not raw_path:
        if mode != DEFAULT_RESUME_MODE:
            raise ValueError(
                "resume_mode requires an explicit resume_checkpoint; "
                "warm starts are never inferred from old checkpoints"
            )
        return None, mode

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"explicit resume checkpoint does not exist: {path}. "
            "Set the path for the intended task, or omit it to use only the "
            "task run directory's normal local resume behavior."
        )
    return path, mode


def get_resume_step(
    train: Mapping[str, Any],
    method_config: Mapping[str, Any] | None,
    checkpoint: Path,
) -> int:
    """Resolve the completed-update number for an explicit checkpoint.

    The checkpoint filename is the normal source of truth.  A separate
    ``resume_step`` is supported for legacy files that were renamed, but it
    must be supplied explicitly rather than guessed from file contents.
    """
    values: dict[str, Any] = dict(train)
    if method_config:
        for key in ("resume_step",):
            if key in method_config:
                values[key] = method_config[key]
    raw_step = str(values.get("resume_step", "") or "").strip()
    if raw_step:
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise ValueError(
                f"resume_step must be an integer, got {raw_step!r} for {checkpoint}"
            ) from exc
    else:
        match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
        if not match:
            raise ValueError(
                f"cannot infer resume step from {checkpoint.name!r}; set "
                "resume_step explicitly in the method config or CLI"
            )
        step = int(match.group(1))
    if step < 0:
        raise ValueError(f"resume_step must be non-negative, got {step}")
    return step


def validate_checkpoint_keys(
    keys: Mapping[str, Any] | set[str] | list[str] | tuple[str, ...],
    mode: str,
    checkpoint: str | Path = "checkpoint",
) -> None:
    """Validate the state shape required by a requested resume mode.

    This is intentionally usable without importing torch, which keeps CLI and
    unit tests lightweight.  The vendored trainer performs the same check on
    the loaded payload immediately before touching model state.
    """
    mode = normalize_resume_mode(mode)
    available = set(keys.keys()) if isinstance(keys, Mapping) else set(keys)
    if mode == "full":
        required = {"net_model", "ema_model", "optim", "sched"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"full-state resume from {checkpoint} is missing {missing}; "
                "the checkpoint must contain net_model, ema_model, optim, and "
                "sched. If it is an EMA-only checkpoint, rerun with explicit "
                "resume_mode=ema_only to opt into a warm start."
            )
    elif mode == "ema_only" and "ema_model" not in available:
        raise ValueError(
            f"EMA-only warm start from {checkpoint} requires key 'ema_model'; "
            f"available keys={sorted(available)}"
        )
