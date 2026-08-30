from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

from .base import Adapter, Phase, resolve_inception_batch_size, resolve_num_workers
from ..checkpoints import get_resume_spec, get_resume_step
from ..config import Task


class CoralAdapter(Adapter):
    name = "coral"

    @staticmethod
    def _flag(name: str, value) -> str:
        if isinstance(value, bool):
            return f"--{name}" if value else f"--no{name}"
        return f"--{name}={value}"

    def _architecture_flags(self, train) -> List[str]:
        """U-Net backbone flags, repeated for absl's multi_integer options."""
        flags: List[str] = []
        if "ch" in train:
            flags.append(self._flag("ch", train["ch"]))
        for name in ("ch_mult", "attn"):
            for value in train.get(name, []):
                flags.append(self._flag(name, value))
        for name in ("num_res_blocks", "ema_decay"):
            if name in train:
                flags.append(self._flag(name, train[name]))
        return flags

    @staticmethod
    def _latest_checkpoint(run_dir: Path, total_steps: int) -> int | None:
        best = None
        for path in run_dir.glob("ckpt_*.pt"):
            match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
            if match and 0 <= int(match.group(1)) < total_steps:
                step = int(match.group(1))
                best = step if best is None else max(best, step)
        return best

    @staticmethod
    def _metrics_root(task: Task) -> Path:
        configured = os.environ.get("LTX_METRICS_ROOT", "").strip()
        if configured:
            root = Path(configured).expanduser()
            if not root.is_absolute():
                # The runner reads .env.local from the campaign root, while
                # Coral phases execute from the vendored repository directory.
                root = Path(task.runtime["repos_root"]).parent / root
            return root.resolve()
        return (Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats").resolve()

    @classmethod
    def _fid_cache(cls, task: Task) -> Path:
        data_type = str(task.dataset.get("data_type", ""))
        cache_name_by_data_type = {
            "cifar10": "cifar10.train.npz",
            "cifar10lt": "cifar10.train.npz",
            "cifar100": "cifar100.train.npz",
            "cifar100lt": "cifar100.train.npz",
        }
        try:
            cache_name = cache_name_by_data_type[data_type]
        except KeyError as exc:
            raise ValueError(
                f"Coral has no canonical FID cache for data_type={data_type!r}; "
                "set an explicit supported CIFAR data_type instead of reusing another dataset's stats"
            ) from exc
        return cls._metrics_root(task) / cache_name

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        train, evaluate = task.train, task.eval
        batch = int(batch_size or train.get("batch_size", 128))
        total = int(train.get("total_steps", 150000))
        inception_batch = resolve_inception_batch_size(evaluate)
        py = task.runtime.get("python", "python")
        fid_cache = self._fid_cache(task)
        metrics_env = {"LTX_METRICS_ROOT": str(fid_cache.parent)}

        common = [
            self._flag("data_type", task.dataset.get("data_type", "cifar10lt")),
            self._flag("imb_factor", task.dataset.get("imbalance_factor", 0.01)),
            self._flag("root", task.dataset.get("root", task.runtime.get("data_root", "./data"))),
            self._flag("logdir", run_dir), self._flag("seed", task.seed), self._flag("fid_cache", fid_cache),
        ]
        if task.dataset.get("frozen_manifest"):
            common.append(self._flag("frozen_manifest", task.dataset["frozen_manifest"]))

        train_cmd = [py, "main.py", "--train", *common,
            self._flag("lr", train.get("lr", 2e-4)), self._flag("batch_size", batch),
            # The Coral source saves the current zero-indexed loop step and
            # uses an inclusive final step in this campaign, so total+1 is
            # intentional: it emits ckpt_<total>.pt for the target budget.
            self._flag("total_steps", total + 1), self._flag("save_step", train.get("save_step", 50000)),
            self._flag("sample_step", train.get("sample_step", 10000)), self._flag("eval_step", train.get("eval_step", 0)),
            self._flag("T", train.get("T", 1000)), self._flag("dropout", train.get("dropout", 0.1)),
            self._flag("num_workers", resolve_num_workers(train, 8)),
        ]
        # Pass the backbone explicitly instead of relying on every repo's flag
        # defaults happening to agree: a vendored-source bump would otherwise
        # change the architecture under a comparison that claims to hold it fixed.
        train_cmd.extend(self._architecture_flags(train))
        if train.get("conditional", True): train_cmd.append("--conditional")
        if train.get("cfg", True): train_cmd.append("--cfg")
        if train.get("amp", False): train_cmd.append("--amp")

        weight_file = task.method_config.get("weight_file", "")
        generated = task.method_config.get("generated_weight", "")
        phases: List[Phase] = []
        if generated:
            weight_file = str(run_dir / f"weights_{generated}.npy")
            prep = [py, str(self.root / "patches" / "prepare_coral_weights.py"),
                    "--repo", str(repo), "--data-type", task.dataset.get("data_type", "cifar10lt"),
                    "--root", task.dataset.get("root", task.runtime.get("data_root", "./data")),
                    "--imb-factor", str(task.dataset.get("imbalance_factor", 0.01)),
                    "--mode", generated, "--output", weight_file]
            if task.dataset.get("frozen_manifest"):
                prep.extend(["--frozen-manifest", task.dataset["frozen_manifest"]])
            phases.append(Phase("prepare_weights", prep, self.root, skip_if_exists=[Path(weight_file), Path(weight_file).with_suffix(".json")]))
        if weight_file:
            train_cmd.append(self._flag("sample_weights", weight_file))
        train_cmd.extend(map(str, task.method_config.get("flags", [])))
        explicit_resume, resume_mode = get_resume_spec(train, task.method_config)
        if explicit_resume is not None:
            resume_step = get_resume_step(train, task.method_config, explicit_resume)
            if resume_step >= total:
                raise ValueError(
                    f"explicit resume checkpoint step {resume_step} must be below "
                    f"the target total_steps {total}: {explicit_resume}"
                )
            train_cmd.extend([
                self._flag("resume_checkpoint", explicit_resume),
                self._flag("ckpt_step", resume_step),
            ])
            if resume_mode == "ema_only":
                # The legacy file contains only EMA weights.  The trainer will
                # initialize a fresh optimizer/scheduler and record this as a
                # non-exact warm start; it must never be mistaken for a
                # bit-exact continuation.
                train_cmd.append("--allow_non_exact_resume")
        else:
            latest = self._latest_checkpoint(run_dir, total)
            if latest is not None:
                # Checkpoints produced by the patched trainer live in this
                # run directory.  The upstream flag defaults
                # ``finetuned_logdir`` to empty, so omitting it makes a rerun
                # fail to load the checkpoint (or encourage a manual fresh
                # start).
                train_cmd.extend([
                    self._flag("resume_checkpoint", run_dir / f"ckpt_{latest}.pt"),
                    self._flag("ckpt_step", latest),
                ])
        phases.append(Phase("train", train_cmd, repo, env=metrics_env,
                            skip_if_exists=[run_dir / f"ckpt_{total}.pt"]))

        scales = task.method_config.get(
            "guidance_scales",
            [task.method_config.get("guidance_scale", evaluate.get("guidance_scale", 1.0))],
        )
        if not isinstance(scales, list): scales = [scales]
        for omega in scales:
            sample_name = f"{task.method}_s{task.seed}_w{omega}"
            suffix = f"{sample_name}_N{evaluate.get('num_images', 50000)}_STEP{total}"
            samples = run_dir / f"{evaluate.get('sample_method','cfg')}_{omega}_samples_ema_{suffix}.npy"
            eval_cmd = [py, "main.py", "--eval", *common,
                self._flag("ckpt_step", total), self._flag("batch_size", batch),
                self._flag("num_images", evaluate.get("num_images", 50000)),
                self._flag("sample_method", evaluate.get("sample_method", "cfg")),
                self._flag("omega", omega), self._flag("sample_name", sample_name),
            ]
            # Added by patches/apply_coral_ddim.py. Passed only when the config
            # asks for it, so an unpatched checkout still runs the ancestral
            # sampler instead of failing on an unknown flag.
            if int(evaluate.get("ddim_steps", 0)) > 0:
                eval_cmd.append(self._flag("ddim_steps", evaluate["ddim_steps"]))
            if train.get("conditional", True): eval_cmd.append("--conditional")
            if evaluate.get("uniform_labels", False): eval_cmd.append("--uniform_labels")
            if evaluate.get("prd", False): eval_cmd.append("--prd")
            if evaluate.get("improved_prd", False): eval_cmd.append("--improved_prd")
            if not evaluate.get("standard_metrics", True): eval_cmd.append("--sample_only")
            phases.append(Phase(f"eval_w{omega}", eval_cmd, repo, env=metrics_env,
                                skip_if_exists=[samples]))
            if evaluate.get("paper_metrics", False):
                labels = Path(str(samples).replace("_samples_", "_labels_"))
                metrics_file = str(evaluate.get("metrics_file", "metrics.paper.json"))
                metric_cmd = [py, str(self.root / "tools" / "evaluate_coral2025.py"),
                    "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"), "--data-type", str(task.dataset["data_type"]),
                    "--samples", str(samples), "--labels", str(labels), "--metrics-root", str(fid_cache.parent),
                    "--output", str(run_dir / metrics_file),
                    "--inception-batch-size", str(inception_batch)]
                if evaluate.get("kid", False):
                    metric_cmd += ["--kid", "--kid-subsets", str(evaluate.get("kid_subsets", 100)),
                                   "--kid-subset-size", str(evaluate.get("kid_subset_size", 1000)),
                                   "--kid-seed", str(evaluate.get("kid_seed", 2026))]
                per_class_file = str(evaluate.get("per_class_metrics_file", "")).strip()
                if per_class_file:
                    metric_cmd += ["--per-class-output", str(run_dir / per_class_file),
                                   "--longtail-groups", str(evaluate.get("longtail_groups", "none"))]
                metric_outputs = [run_dir / metrics_file]
                if per_class_file:
                    metric_outputs.append(run_dir / per_class_file)
                phases.append(Phase(f"paper_metrics_w{omega}", metric_cmd, self.root, env=metrics_env,
                                    skip_if_exists=metric_outputs))

        if task.semantic_eval_command:
            omega = scales[-1]
            sample_name = f"{task.method}_s{task.seed}_w{omega}_N{evaluate.get('num_images',50000)}_STEP{total}"
            prefix = evaluate.get("sample_method", "cfg")
            samples = run_dir / f"{prefix}_{omega}_samples_ema_{sample_name}.npy"
            labels = run_dir / f"{prefix}_{omega}_labels_ema_{sample_name}.npy"
            output = run_dir / "semantic_metrics.json"
            rendered = task.semantic_eval_command.format(
                samples=samples, labels=labels, run_dir=run_dir, output=output,
                manifest=task.dataset.get("frozen_manifest", ""), method=task.method, seed=task.seed,
            )
            phases.append(Phase("semantic_eval", ["bash", "-lc", rendered], self.root, skip_if_exists=[output]))

        (run_dir / "task.json").write_text(json.dumps(task.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return phases
