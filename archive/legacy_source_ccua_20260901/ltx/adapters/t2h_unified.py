from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable, List

import torch

from .base import Adapter, Phase, resolve_inception_batch_size, resolve_num_workers
from ..checkpoints import get_resume_spec, get_resume_step
from ..config import Task


class T2HUnifiedAdapter(Adapter):
    """Run every benchmark row through the T2H/OC_LT host.

    ``method`` selects only the objective dispatch inside ``unified_main.py``.
    The model, corruption schedule, loader, checkpoint format, sampler, and
    output contract are therefore shared by all rows.  The old native repos
    are deliberately not referenced by these phases.
    """

    name = "t2h_unified"
    # A new namespace makes old native/mixed artifacts in an existing run
    # directory ineligible for automatic reuse.  The host also validates the
    # embedded provenance, so renaming is a first-line guard rather than the
    # only correctness check.
    checkpoint_prefix = "ckpt_unified_v2_"
    sample_namespace = "t2h_unified_v2"
    host_revision = "t2h-unified-common-v2"
    checkpoint_schema = 2

    @staticmethod
    def _latest(run_dir: Path, target: int, prefix: str) -> int | None:
        best = None
        pattern = re.compile(re.escape(prefix) + r"(\d+)\.pt")
        for path in run_dir.glob(f"{prefix}*.pt"):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            step = int(match.group(1))
            if step < target and (best is None or step > best):
                best = step
        return best

    @staticmethod
    def _value(flags: Iterable[object], name: str, default):
        prefix = f"--{name}="
        for flag in flags:
            value = str(flag)
            if value == f"--{name}":
                return True
            if value.startswith(prefix):
                raw = value[len(prefix):]
                try:
                    return type(default)(raw)
                except (TypeError, ValueError):
                    return raw
        return default

    @staticmethod
    def _host_objective(method: str) -> str:
        method = method.lower()
        if method in {"t2h", "oc"}:
            return "t2h"
        if method in {"ipsvt", "ipsvt_twin", "ipsvt_clean", "ipsvt_response_twin", "ipsvt_response_full", "ipsvt_hybrid"}:
            return "ipsvt"
        return method

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _sample_provenance_valid(cls, path: Path, expected: dict[str, Any]) -> bool:
        actual = cls._read_json(path)
        return actual is not None and cls._provenance_values_match(actual, expected)

    @classmethod
    def _metric_provenance_valid(
        cls,
        path: Path,
        expected: dict[str, Any],
        metric_host: str,
        *,
        require_kid: bool = True,
    ) -> bool:
        actual = cls._read_json(path)
        if actual is None:
            return False
        provenance = actual.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("metric_host") != metric_host:
            return False
        sample = provenance.get("sample")
        if not isinstance(sample, dict) or not cls._provenance_values_match(sample, expected):
            return False
        if metric_host == "common_cifar_metrics_v2":
            metrics = actual.get("metrics")
            if isinstance(metrics, dict):
                # The evaluator writes a fast headline snapshot before the
                # VGG16 improved-PRD pass.  It is not a completed metric phase
                # until the detailed columns are present too.
                required = {
                    "FID", "IS", "F_8", "F_1_8",
                    "ImprovedPrecision", "Recall",
                }
                # Detailed unified metrics may deliberately omit KID for a
                # diagnostic smoke.  Its requested presence is configuration
                # provenance, not a universal completion requirement.
                if require_kid:
                    required.add("KID")
                return required.issubset(metrics)
            per_class = actual.get("per_class")
            groups = actual.get("groups")
            return (
                isinstance(per_class, dict)
                and isinstance(groups, dict)
                and {"Many", "Medium", "Few"}.issubset(groups)
            )
        if metric_host == "common_imagenet_metrics_v2":
            return "FID" in actual and isinstance(actual.get("KID"), dict)
        return False

    @staticmethod
    def _provenance_values_match(actual: Any, expected: Any) -> bool:
        """Compare JSON provenance while tolerating harmless float round-off."""
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and T2HUnifiedAdapter._provenance_values_match(actual[key], value)
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return isinstance(actual, list) and len(actual) == len(expected) and all(
                T2HUnifiedAdapter._provenance_values_match(left, right)
                for left, right in zip(actual, expected)
            )
        if isinstance(expected, float):
            try:
                left = float(actual)
            except (TypeError, ValueError):
                return False
            return abs(left - expected) <= 1e-12 * max(1.0, abs(left), abs(expected))
        return actual == expected

    @classmethod
    def _expected_host_provenance(cls, task: Task, batch: int) -> dict[str, Any]:
        """Build the host manifest identity from the adapter's effective flags.

        The host writes this same structure into every checkpoint.  Checking it
        before automatic resume is important: a stale ``unified_host.json`` or
        a checkpoint from a different objective must not be selected merely
        because its filename happens to be the newest one.
        """
        train = task.train
        method = task.method.lower()
        host_objective = cls._host_objective(method)
        cfg = task.method_config
        source_flags = cfg.get("flags", [])

        def effective(key: str, flag_name: str, default):
            return cfg.get(key, cls._value(source_flags, flag_name, default))

        objective_config: dict[str, Any] = {
            "cb_tau": 1.0,
            "coral_weight": 0.01,
            "coral_temperature": 0.09,
            "coral_temperature_scaling": 1.0,
            "ccua_al": 0.0,
            "ccua_ucl": 0.0,
            "cm_w_con": 1.0,
            "cm_w_div": 0.2,
            "cm_lora_r": 0,
            "cm_lora_alpha": 1.0,
            "cm_lora_r_ratio": 0.1,
            "cm_lora_scaling": 0.5,
            "cm_lora_mode": "ratio",
            "ipsvt_mode": "full",
            "ipsvt_lambda_aux": 1.0,
            "ipsvt_lambda_svt": 1.0,
            "ipsvt_K": 4,
            "ipsvt_s": 0.05,
            "ipsvt_delta": 0.1,
            "ipsvt_every": 4,
            "ipsvt_batch": 16,
            "transfer_x0": host_objective in {"t2h"} or method == "cm",
            "transfer_mode": "t2h",
            "t2h_cut_time": -1,
        }
        if method == "cbdm":
            objective_config["cb_tau"] = effective("cb_tau", "tau", 1.0)
        elif method == "coral":
            objective_config.update({
                "coral_weight": effective("supcon_weight", "supcon_weight", 0.01),
                "coral_temperature": effective("supcon_temp", "supcon_temp", 0.09),
                "coral_temperature_scaling": effective("temperature_scaling", "temperature_scaling", 1.0),
            })
        elif method == "ccua":
            # The adapter passes 1/1 even when the method stanza omits both
            # values, matching unified_main.py's effective CCUA defaults.
            objective_config.update({
                "ccua_al": cfg.get("ccua_al", 1.0),
                "ccua_ucl": cfg.get("ccua_ucl", 1.0),
            })
        elif method == "cm":
            objective_config.update({
                "cm_w_con": cfg.get("w_con", 1.0),
                "cm_w_div": cfg.get("w_div", 0.2),
                "cm_lora_r": cfg.get("lora_r", 0),
                "cm_lora_alpha": cfg.get("lora_alpha", 1.0),
                "cm_lora_r_ratio": cfg.get("lora_r_ratio", 0.1),
                "cm_lora_scaling": cfg.get("lora_scaling", 0.5),
                "cm_lora_mode": cfg.get("lora_mode", "ratio"),
            })
        elif method in {"ipsvt", "ipsvt_twin", "ipsvt_clean"}:
            default_mode = "clean" if method == "ipsvt_clean" else (
                "twin" if method == "ipsvt_twin" else "full"
            )
            objective_config.update({
                "ipsvt_mode": cfg.get("ipsvt_mode", cls._value(source_flags, "ipsvt_mode", default_mode)),
                "ipsvt_lambda_aux": cfg.get("ipsvt_lambda_aux", cls._value(source_flags, "ipsvt_lambda_aux", 1.0)),
                "ipsvt_lambda_svt": cfg.get("ipsvt_lambda_svt", cls._value(source_flags, "ipsvt_lambda_svt", 1.0)),
                "ipsvt_K": cfg.get("ipsvt_K", cls._value(source_flags, "ipsvt_K", 4)),
                "ipsvt_s": cfg.get("ipsvt_s", cls._value(source_flags, "ipsvt_s", 0.05)),
                "ipsvt_delta": cfg.get("ipsvt_delta", cls._value(source_flags, "ipsvt_delta", 0.1)),
                "ipsvt_every": cfg.get("ipsvt_every", cls._value(source_flags, "ipsvt_every", 4)),
                "ipsvt_batch": cfg.get("ipsvt_batch", cls._value(source_flags, "ipsvt_batch", 16)),
            })
        elif method in {"ipsvt_response_twin", "ipsvt_response_full"}:
            variant = "twin" if method == "ipsvt_response_twin" else "full"
            objective_config.update({
                "ipsvt_mode": "response",
                "ipsvt_response_variant": cfg.get("ipsvt_response_variant", variant),
                "ipsvt_response_eta": cfg.get("ipsvt_response_eta", 0.05),
                "ipsvt_lambda": cfg.get("ipsvt_lambda", 1.0),
            })
        elif method == "ipsvt_hybrid":
            objective_config.update({
                "ipsvt_mode": "hybrid",
                "ipsvt_lambda_aux": cfg.get("ipsvt_lambda_aux", 1.0),
                "ipsvt_lambda_svt": cfg.get("ipsvt_lambda_svt", 1.0),
                "ipsvt_K": cfg.get("ipsvt_K", 4),
                "ipsvt_s": cfg.get("ipsvt_s", 0.05),
                "ipsvt_delta": cfg.get("ipsvt_delta", 0.1),
                "ipsvt_tau": cfg.get("ipsvt_tau", 1e-6),
                "ipsvt_hybrid_chunk": cfg.get("ipsvt_hybrid_chunk", 16),
            })
            # These two keys belong only to legacy IPSVTAuxiliary scheduling;
            # hybrid has no auxiliary sampler or every-N-step gate.
            objective_config.pop("ipsvt_every")
            objective_config.pop("ipsvt_batch")

        num_class = int(task.dataset.get("num_class", task.dataset.get("num_classes", 100)))
        img_size = int(task.dataset.get("img_size", 32))
        return {
            "schema": cls.checkpoint_schema,
            "host": "T2H-unified",
            "host_revision": cls.host_revision,
            "objective": host_objective,
            "data": {
                "data_type": str(task.dataset.get("data_type", "cifar100lt")).lower(),
                "imb_factor": float(task.dataset.get("imbalance_factor", 0.01)),
                "split_seed": int(task.dataset.get("split_seed", 0)),
                "num_class": num_class,
                "img_size": img_size,
            },
            "model": {
                "T": int(train.get("T", 1000)),
                "ch": int(train.get("ch", 128)),
                "ch_mult": list(train.get("ch_mult", [1, 2, 2, 2])),
                "attn": list(train.get("attn", [1])),
                "num_res_blocks": int(train.get("num_res_blocks", 2)),
                "dropout": float(train.get("dropout", 0.1)),
                "conditional": bool(train.get("conditional", True)),
                "coral_projection_dim": int(cfg.get("coral_projection_dim", 128))
                if method == "coral" or method in {"ipsvt_response_twin", "ipsvt_response_full", "ipsvt_hybrid"} else 0,
                "cm_lora_part": list(cfg.get("lora_part", ["up"])) if method == "cm" else [],
            },
            "training": {
                "seed": int(task.seed),
                "batch_size": int(batch),
                "lr": float(train.get("lr", 2e-4)),
                "warmup": int(train.get("warmup", 5000)),
                "grad_clip": float(train.get("grad_clip", 1.0)),
                "ema_decay": float(train.get("ema_decay", 0.9999)),
                "cfg": bool(train.get("cfg", True)),
                "amp": bool(train.get("amp", False)),
            },
            "diffusion": {
                "beta_1": float(train.get("beta_1", 1e-4)),
                "beta_T": float(train.get("beta_T", 0.02)),
                "var_type": str(train.get("var_type", "fixedlarge")),
            },
            "objective_config": objective_config,
        }

    @classmethod
    def _train_manifest_valid(cls, path: Path, task: Task, batch: int, target: int) -> bool:
        actual = cls._read_json(path)
        if actual is None:
            return False
        try:
            num_class = int(task.dataset.get("num_class", task.dataset.get("num_classes", 100)))
            expected_bound = int(task.train.get("total_steps", target)) + int(
                bool(task.train.get("inclusive_final_step", True))
            )
            actual_seed = int(actual.get("seed", -1))
            actual_num_class = int(actual.get("num_class", -1))
            actual_bound = int(actual.get("total_steps_bound", -1))
        except (TypeError, ValueError, OverflowError):
            return False
        expected = cls._expected_host_provenance(task, batch)
        return (
            actual.get("host") == "T2H-unified"
            and actual.get("host_revision") == cls.host_revision
            and actual.get("checkpoint_schema") == cls.checkpoint_schema
            and actual.get("objective") == expected["objective"]
            and actual_seed == int(task.seed)
            and str(actual.get("data_type", "")).lower() == expected["data"]["data_type"]
            and actual_num_class == num_class
            and actual_bound == expected_bound
            and cls._provenance_values_match(actual.get("provenance"), expected)
        )

    @classmethod
    def _checkpoint_valid(cls, path: Path, task: Task, batch: int, step: int) -> bool:
        """Check the local checkpoint before using it for skip/auto-resume.

        The host performs the authoritative validation immediately before
        loading tensors.  This cheaper adapter-side check prevents a corrupt,
        truncated, or EMA-only file from making a phase look complete merely
        because its filename and host manifest exist.
        """
        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # PyTorch versions before the weights_only kwarg
                checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            return False
        if not isinstance(checkpoint, dict):
            return False
        try:
            completed_step = int(checkpoint.get("step", -1))
        except (TypeError, ValueError, OverflowError):
            return False
        if completed_step != int(step):
            return False
        if not {"net_model", "ema_model", "optim", "sched"}.issubset(checkpoint):
            return False
        expected = cls._expected_host_provenance(task, batch)
        return cls._provenance_values_match(checkpoint.get("provenance"), expected)

    def _objective_flags(self, task: Task) -> List[str]:
        method = task.method.lower()
        cfg = task.method_config
        source_flags = cfg.get("flags", [])
        if method == "ddpm":
            return ["--objective=ddpm"]
        if method in {"t2h", "oc"}:
            return ["--objective=t2h"]
        if method == "cbdm":
            tau = cfg.get("cb_tau", self._value(source_flags, "tau", 1.0))
            return ["--objective=cbdm", f"--cb_tau={tau}"]
        if method == "coral":
            weight = cfg.get("supcon_weight", self._value(source_flags, "supcon_weight", 0.01))
            temperature = cfg.get("supcon_temp", self._value(source_flags, "supcon_temp", 0.09))
            scaling = cfg.get("temperature_scaling", self._value(source_flags, "temperature_scaling", 1.0))
            return ["--objective=coral", f"--coral_weight={weight}",
                    f"--coral_temperature={temperature}",
                    f"--coral_temperature_scaling={scaling}",
                    f"--coral_projection_dim={cfg.get('coral_projection_dim', 128)}"]
        if method == "ccua":
            return ["--objective=ccua",
                    f"--ccua_al={cfg.get('ccua_al', 1.0)}",
                    f"--ccua_ucl={cfg.get('ccua_ucl', 1.0)}"]
        if method == "cm":
            return ["--objective=cm",
                    f"--cm_w_con={cfg.get('w_con', 1.0)}",
                    f"--cm_w_div={cfg.get('w_div', 0.2)}",
                    f"--cm_lora_r={cfg.get('lora_r', 0)}",
                    f"--cm_lora_alpha={cfg.get('lora_alpha', 1.0)}",
                    f"--cm_lora_r_ratio={cfg.get('lora_r_ratio', 0.1)}",
                    f"--cm_lora_scaling={cfg.get('lora_scaling', 0.5)}",
                    f"--cm_lora_mode={cfg.get('lora_mode', 'ratio')}"] + [
                        f"--cm_lora_part={part}" for part in cfg.get('lora_part', ['up'])
                    ]
        if method in {"ipsvt", "ipsvt_twin", "ipsvt_clean"}:
            mode = "clean" if method == "ipsvt_clean" else ("twin" if method == "ipsvt_twin" else "full")
            mode = cfg.get("ipsvt_mode", self._value(source_flags, "ipsvt_mode", mode))
            values = {
                "ipsvt_K": cfg.get("ipsvt_K", self._value(source_flags, "ipsvt_K", 4)),
                "ipsvt_s": cfg.get("ipsvt_s", self._value(source_flags, "ipsvt_s", 0.05)),
                "ipsvt_delta": cfg.get("ipsvt_delta", self._value(source_flags, "ipsvt_delta", 0.1)),
                "ipsvt_every": cfg.get("ipsvt_every", self._value(source_flags, "ipsvt_every", 4)),
                "ipsvt_batch": cfg.get("ipsvt_batch", self._value(source_flags, "ipsvt_batch", 16)),
                "ipsvt_lambda_aux": cfg.get("ipsvt_lambda_aux", self._value(source_flags, "ipsvt_lambda_aux", 1.0)),
                "ipsvt_lambda_svt": cfg.get("ipsvt_lambda_svt", self._value(source_flags, "ipsvt_lambda_svt", 1.0)),
            }
            return ["--objective=ipsvt", "--ipsvt", f"--ipsvt_mode={mode}"] + [
                f"--{key}={value}" for key, value in values.items()
            ]
        if method in {"ipsvt_response_twin", "ipsvt_response_full"}:
            variant = "twin" if method == "ipsvt_response_twin" else "full"
            variant = cfg.get("ipsvt_response_variant", variant)
            eta = cfg.get("ipsvt_response_eta", 0.05)
            weight = cfg.get("ipsvt_lambda", 1.0)
            return [
                "--objective=ipsvt", "--ipsvt", "--ipsvt_mode=response",
                f"--ipsvt_response_variant={variant}",
                f"--ipsvt_response_eta={eta}", f"--ipsvt_lambda={weight}",
                # Strict native import needs the otherwise-unused projection
                # tensors present in the source DDPM checkpoint.
                f"--coral_projection_dim={cfg.get('coral_projection_dim', 128)}",
            ]
        if method == "ipsvt_hybrid":
            values = {
                "ipsvt_K": cfg.get("ipsvt_K", 4),
                "ipsvt_s": cfg.get("ipsvt_s", 0.05),
                "ipsvt_delta": cfg.get("ipsvt_delta", 0.1),
                "ipsvt_tau": cfg.get("ipsvt_tau", 1e-6),
                "ipsvt_lambda_aux": cfg.get("ipsvt_lambda_aux", 1.0),
                "ipsvt_lambda_svt": cfg.get("ipsvt_lambda_svt", 1.0),
                "ipsvt_hybrid_chunk": cfg.get("ipsvt_hybrid_chunk", 16),
            }
            return [
                "--objective=ipsvt", "--ipsvt", "--ipsvt_mode=hybrid",
                f"--coral_projection_dim={cfg.get('coral_projection_dim', 128)}",
                *[f"--{key}={value}" for key, value in values.items()],
            ]
        raise ValueError(f"T2H unified host does not know method={task.method!r}")

    @staticmethod
    def _architecture_flags(train: dict) -> List[str]:
        flags: List[str] = []
        if "ch" in train:
            flags.append(f"--ch={train['ch']}")
        for key in ("ch_mult", "attn"):
            flags.extend(f"--{key}={value}" for value in train.get(key, []))
        for key in ("num_res_blocks", "ema_decay"):
            if key in train:
                flags.append(f"--{key}={train[key]}")
        return flags

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        train, evaluate = task.train, task.eval
        py = task.runtime.get("python", "python")
        batch = int(batch_size or train.get("batch_size", 64))
        target = int(evaluate.get("checkpoint_step", train.get("total_steps", 300000)))
        total_bound = int(train.get("total_steps", target)) + int(bool(train.get("inclusive_final_step", True)))
        num_class = task.dataset.get("num_class", task.dataset.get("num_classes", 100))
        img_size = task.dataset.get("img_size", 32)
        data_type = task.dataset.get("data_type", "cifar100lt")
        data_root = task.dataset.get("root", task.runtime.get("data_root", "./data"))
        checkpoint_prefix = str(train.get("checkpoint_prefix", self.checkpoint_prefix))
        artifact_namespace = str(evaluate.get("artifact_namespace", self.sample_namespace))
        if not checkpoint_prefix.endswith("_"):
            raise ValueError("T2H unified checkpoint_prefix must end with '_' to avoid ambiguous step names")
        target_checkpoint = run_dir / f"{checkpoint_prefix}{target}.pt"
        host_manifest = run_dir / "unified_host.json"
        explicit_resume, resume_mode = get_resume_spec(train, task.method_config)
        import_checkpoint = str(task.method_config.get("import_checkpoint", "")).strip()
        if import_checkpoint:
            path = Path(import_checkpoint).expanduser()
            if not path.is_absolute():
                path = (self.root / path).resolve()
            import_checkpoint = str(path)
        import_step = task.method_config.get("import_checkpoint_step", None)
        import_sha256 = str(task.method_config.get("import_checkpoint_sha256", "")).strip()
        if import_checkpoint and import_step is None:
            raise ValueError("T2H native smoke import requires method.import_checkpoint_step")
        if import_checkpoint and explicit_resume is not None:
            raise ValueError("T2H native smoke import cannot be combined with resume_checkpoint")
        import_flags: List[str] = []
        if import_checkpoint:
            import_flags = [
                f"--import_checkpoint={import_checkpoint}",
                f"--import_checkpoint_step={int(import_step)}",
                "--allow_legacy_resume",
            ]
            if import_sha256:
                import_flags.append(f"--import_checkpoint_sha256={import_sha256}")

        common = [
            py, "unified_main.py", "--train", *self._objective_flags(task),
            f"--data_type={data_type}", f"--root={data_root}",
            f"--imb_factor={task.dataset.get('imbalance_factor', 0.01)}",
            f"--split_seed={task.dataset.get('split_seed', 0)}",
            f"--num_class={num_class}", f"--img_size={img_size}",
            f"--logdir={run_dir}", f"--seed={task.seed}",
            f"--checkpoint_prefix={checkpoint_prefix}",
            f"--total_steps={total_bound}", f"--batch_size={batch}",
            f"--lr={train.get('lr', 2e-4)}", f"--warmup={train.get('warmup', 5000)}",
            f"--T={train.get('T', 1000)}",
            f"--beta_1={train.get('beta_1', 1e-4)}", f"--beta_T={train.get('beta_T', 0.02)}",
            f"--var_type={train.get('var_type', 'fixedlarge')}",
            f"--dropout={train.get('dropout', 0.1)}",
            f"--grad_clip={train.get('grad_clip', 1.0)}",
            f"--num_workers={resolve_num_workers(train, 4)}",
            f"--save_step={train.get('save_step', 50000)}",
            f"--sample_step={train.get('sample_step', 100000)}",
        ]
        if train.get("conditional", True):
            common.append("--conditional")
        if train.get("cfg", True):
            common.append("--cfg")
        if train.get("amp", False):
            common.append("--amp")
        common.extend(self._architecture_flags(train))
        if task.dataset.get("frozen_manifest"):
            common.append(f"--frozen_manifest={task.dataset['frozen_manifest']}")
        if task.dataset.get("manifest"):
            common.append(f"--train_manifest={task.dataset['manifest']}")

        # These flags carry the native source identity into a verified T2H
        # resume/sample too. unified_main.py loads native tensors only when no
        # T2H --resume_checkpoint is selected; otherwise it validates the same
        # source hash as part of the continuation provenance.
        common.extend(import_flags)

        if explicit_resume is not None:
            if resume_mode != "full":
                raise ValueError(
                    "T2H unified host accepts only full-state resume checkpoints; "
                    "its provenance guard intentionally rejects unverified EMA-only files"
                )
            # Do not let a local automatic-resume checkpoint override the
            # explicit lineage selected by the CLI/config.  unified_main.py
            # validates its embedded provenance before restoring state.
            common.append(f"--resume_checkpoint={explicit_resume}")
            common.append(
                f"--resume_step={get_resume_step(train, task.method_config, explicit_resume)}"
            )
        else:
            latest = self._latest(run_dir, target, checkpoint_prefix)
            if (
                latest is not None
                and self._train_manifest_valid(host_manifest, task, batch, target)
                and self._checkpoint_valid(
                    run_dir / f"{checkpoint_prefix}{latest}.pt", task, batch, latest
                )
            ):
                common.append(f"--resume_checkpoint={run_dir / f'{checkpoint_prefix}{latest}.pt'}")
        phases = [Phase(
            "train", common, repo,
            skip_if_exists=[target_checkpoint, host_manifest],
            skip_if_valid=lambda: (
                explicit_resume is None
                and self._train_manifest_valid(host_manifest, task, batch, target)
                and self._checkpoint_valid(target_checkpoint, task, batch, target)
            ),
        )]

        scales = task.method_config.get(
            "guidance_scales",
            [task.method_config.get("guidance_scale", evaluate.get("guidance_scale", 1.5))],
        )
        if not isinstance(scales, list):
            scales = [scales]
        single = len(scales) == 1
        inception_batch = resolve_inception_batch_size(evaluate)
        for omega in scales:
            suffix = "" if single else f"_w{omega}"
            samples = run_dir / f"samples.{artifact_namespace}{suffix}.npy"
            labels = Path(str(samples) + ".labels.npy")
            sample_provenance = samples.with_suffix(".provenance.json")
            marker = run_dir / f"T2H_UNIFIED_SAMPLE_V2_DONE{suffix}"
            sample_method = str(evaluate.get("sample_method", "ddim"))
            ddim_skip_step = int(evaluate.get("ddim_skip_step", 10))
            sampler_method = "ddim" if sample_method in {"ddim", "cfg"} else "ddpm"
            expected_sample_provenance = {
                "host": "T2H-unified",
                "objective": self._host_objective(task.method),
                "host_revision": self.host_revision,
                "checkpoint_schema": self.checkpoint_schema,
                "checkpoint_step": target,
                "num_images": int(evaluate.get("num_images", 50000)),
                "sample_method": sample_method,
                "sampler_method": sampler_method,
                "ddim_skip_step": ddim_skip_step,
                "omega": float(omega),
                "uniform_labels": bool(evaluate.get("uniform_labels", False)),
                "seed": int(task.seed),
                "artifact_namespace": artifact_namespace,
                "T": int(train.get("T", 1000)),
                "beta_1": float(train.get("beta_1", 1e-4)),
                "beta_T": float(train.get("beta_T", 0.02)),
                "var_type": str(train.get("var_type", "fixedlarge")),
                "img_size": int(img_size),
                "num_class": int(num_class),
            }
            sample_cmd = [
                py, "unified_main.py", "--sample", *self._objective_flags(task),
                f"--data_type={data_type}", f"--root={data_root}",
                f"--imb_factor={task.dataset.get('imbalance_factor', 0.01)}",
                f"--split_seed={task.dataset.get('split_seed', 0)}",
                f"--num_class={num_class}", f"--img_size={img_size}",
                f"--logdir={run_dir}", f"--seed={task.seed}",
                f"--checkpoint_prefix={checkpoint_prefix}",
                f"--ckpt_step={target}", f"--num_images={evaluate.get('num_images', 50000)}",
                f"--sample_batch_size={evaluate.get('sample_batch_size', batch)}",
                f"--sample_method={evaluate.get('sample_method', 'ddim')}",
                f"--ddim_skip_step={evaluate.get('ddim_skip_step', 10)}",
                f"--omega={omega}", f"--sample_output={samples}",
                f"--artifact_namespace={artifact_namespace}",
                f"--batch_size={batch}", f"--lr={train.get('lr', 2e-4)}",
                f"--warmup={train.get('warmup', 5000)}",
                f"--dropout={train.get('dropout', 0.1)}",
                f"--grad_clip={train.get('grad_clip', 1.0)}",
                f"--ema_decay={train.get('ema_decay', 0.9999)}",
                f"--T={train.get('T', 1000)}",
                f"--beta_1={train.get('beta_1', 1e-4)}", f"--beta_T={train.get('beta_T', 0.02)}",
                f"--coral_projection_dim={task.method_config.get('coral_projection_dim', 128)}",
                f"--var_type={train.get('var_type', 'fixedlarge')}",
                *import_flags,
            ]
            sample_cmd.extend(self._architecture_flags(train))
            if train.get("conditional", True):
                sample_cmd.append("--conditional")
            if train.get("cfg", True):
                sample_cmd.append("--cfg")
            if train.get("amp", False):
                sample_cmd.append("--amp")
            if evaluate.get("uniform_labels", False):
                sample_cmd.append("--uniform_labels")
            if task.dataset.get("frozen_manifest"):
                sample_cmd.append(f"--frozen_manifest={task.dataset['frozen_manifest']}")
            if task.dataset.get("manifest"):
                sample_cmd.append(f"--train_manifest={task.dataset['manifest']}")
            quoted = " ".join(shlex.quote(str(x)) for x in sample_cmd)
            phases.append(Phase(
                f"sample{suffix}", ["bash", "-lc", f"{quoted} && touch {shlex.quote(str(marker))}"],
                repo, skip_if_exists=[marker, samples, labels, sample_provenance],
                skip_if_valid=lambda path=sample_provenance, expected=expected_sample_provenance:
                    self._sample_provenance_valid(path, expected),
            ))

            if not evaluate.get("paper_metrics", False):
                continue
            metrics_name = str(evaluate.get("metrics_file", "metrics.unified.json"))
            if not single:
                metrics_name = metrics_name.replace(".json", f"{suffix}.json")
            metrics = run_dir / metrics_name
            expected_metric_flags = [
                "--expected-host-revision", self.host_revision,
                "--expected-checkpoint-schema", str(self.checkpoint_schema),
                "--expected-objective", self._host_objective(task.method),
                "--expected-checkpoint-step", str(target),
                "--expected-num-images", str(expected_sample_provenance["num_images"]),
                "--expected-sample-method", sample_method,
                "--expected-sampler-method", sampler_method,
                "--expected-ddim-skip-step", str(ddim_skip_step),
                "--expected-omega", str(float(omega)),
                "--expected-seed", str(task.seed),
                "--expected-artifact-namespace", artifact_namespace,
                "--expected-T", str(expected_sample_provenance["T"]),
                "--expected-beta-1", str(expected_sample_provenance["beta_1"]),
                "--expected-beta-T", str(expected_sample_provenance["beta_T"]),
                "--expected-var-type", str(expected_sample_provenance["var_type"]),
                "--expected-img-size", str(expected_sample_provenance["img_size"]),
                "--expected-num-class", str(expected_sample_provenance["num_class"]),
            ]
            if expected_sample_provenance["uniform_labels"]:
                expected_metric_flags.append("--expected-uniform-labels")
            if str(data_type).lower().startswith("imagenet"):
                reference = task.dataset.get("reference_manifest", "")
                if not reference:
                    raise ValueError("ImageNet-LT requires dataset.reference_manifest")
                metric_cmd = [
                    py, str(self.root / "tools" / "evaluate_imagenet_lt.py"),
                    "--repo", str(repo),
                    "--image-root", str(data_root), "--reference-manifest", str(reference),
                    "--samples", str(samples), "--labels", str(labels),
                    "--num-images", str(evaluate.get("num_images", 50000)),
                    "--image-size", str(img_size), "--num-classes", str(num_class),
                    "--batch-size", str(inception_batch), "--seed", str(task.seed),
                    "--kid-repeats", str(evaluate.get("kid_repeats", 2)),
                    "--weights", str(repo / "stats" / "pt_inception-2015-12-05-6726825d.pth"),
                    "--output", str(metrics),
                    *expected_metric_flags,
                ]
            else:
                metric_cmd = [
                    py, str(self.root / "tools" / "evaluate_coral2025.py"),
                    "--repo", str(repo), "--data-type", str(data_type),
                    "--samples", str(samples), "--labels", str(labels),
                    "--metrics-root", str(repo / "stats"),
                    "--output", str(metrics), "--inception-batch-size", str(inception_batch),
                    "--vgg-batch-size", str(evaluate.get("vgg_batch_size", 128)),
                    "--knn-query-batch", str(evaluate.get("knn_query_batch", 1024)),
                    *expected_metric_flags,
                ]
                # The evaluator's default is detailed, but a diagnostic can
                # pin it explicitly so its required tail-FID and VGG PRD
                # outputs cannot silently fall back to headline mode.
                if "metrics_mode" in evaluate:
                    metric_cmd += ["--mode", str(evaluate["metrics_mode"])]
                if evaluate.get("kid", False):
                    metric_cmd += ["--kid", "--kid-subsets", str(evaluate.get("kid_subsets", 100)),
                                   "--kid-subset-size", str(evaluate.get("kid_subset_size", 1000)),
                                   "--kid-seed", str(evaluate.get("kid_seed", 2026))]
                per_class = str(evaluate.get("per_class_metrics_file", "")).strip()
                if per_class and single:
                    metric_cmd += ["--per-class-output", str(run_dir / per_class),
                                   "--longtail-groups", str(evaluate.get("longtail_groups", "none"))]
            outputs = [metrics]
            per_class = str(evaluate.get("per_class_metrics_file", "")).strip()
            if per_class and single and not str(data_type).lower().startswith("imagenet"):
                outputs.append(run_dir / per_class)
            metric_host = (
                "common_imagenet_metrics_v2"
                if str(data_type).lower().startswith("imagenet")
                else "common_cifar_metrics_v2"
            )
            phases.append(Phase(
                f"metrics{suffix}", metric_cmd, self.root, skip_if_exists=outputs,
                skip_if_valid=lambda paths=tuple(outputs), expected=expected_sample_provenance,
                host=metric_host, require_kid=bool(evaluate.get("kid", False)):
                    all(self._metric_provenance_valid(path, expected, host, require_kid=require_kid)
                        for path in paths),
            ))

        return phases
