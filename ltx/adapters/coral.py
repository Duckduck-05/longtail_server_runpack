from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from .base import Adapter, Phase, resolve_num_workers
from ..config import Task


class CoralAdapter(Adapter):
    name = "coral"

    @staticmethod
    def _flag(name: str, value) -> str:
        if isinstance(value, bool):
            return f"--{name}" if value else f"--no{name}"
        return f"--{name}={value}"

    @staticmethod
    def _latest_checkpoint(run_dir: Path, total_steps: int) -> int:
        best = 0
        for path in run_dir.glob("ckpt_*.pt"):
            match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
            if match and 0 < int(match.group(1)) < total_steps:
                best = max(best, int(match.group(1)))
        return best

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        train, evaluate = task.train, task.eval
        batch = int(batch_size or train.get("batch_size", 128))
        total = int(train.get("total_steps", 150000))
        py = task.runtime.get("python", "python")

        common = [
            self._flag("data_type", task.dataset.get("data_type", "cifar10lt")),
            self._flag("imb_factor", task.dataset.get("imbalance_factor", 0.01)),
            self._flag("root", task.dataset.get("root", task.runtime.get("data_root", "./data"))),
            self._flag("logdir", run_dir), self._flag("seed", task.seed),
        ]
        if task.dataset.get("frozen_manifest"):
            common.append(self._flag("frozen_manifest", task.dataset["frozen_manifest"]))

        train_cmd = [py, "main.py", "--train", *common,
            self._flag("lr", train.get("lr", 2e-4)), self._flag("batch_size", batch),
            self._flag("total_steps", total + 1), self._flag("save_step", train.get("save_step", 50000)),
            self._flag("sample_step", train.get("sample_step", 10000)), self._flag("eval_step", train.get("eval_step", 0)),
            self._flag("T", train.get("T", 1000)), self._flag("dropout", train.get("dropout", 0.1)),
            self._flag("num_workers", resolve_num_workers(train, 8)),
        ]
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
        latest = self._latest_checkpoint(run_dir, total)
        if latest:
            train_cmd.append(self._flag("ckpt_step", latest))
        phases.append(Phase("train", train_cmd, repo, skip_if_exists=[run_dir / f"ckpt_{total}.pt"]))

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
            if train.get("conditional", True): eval_cmd.append("--conditional")
            if evaluate.get("uniform_labels", False): eval_cmd.append("--uniform_labels")
            if evaluate.get("prd", False): eval_cmd.append("--prd")
            if evaluate.get("improved_prd", False): eval_cmd.append("--improved_prd")
            if not evaluate.get("standard_metrics", True): eval_cmd.append("--sample_only")
            phases.append(Phase(f"eval_w{omega}", eval_cmd, repo, skip_if_exists=[samples]))
            if evaluate.get("paper_metrics", False):
                labels = Path(str(samples).replace("_samples_", "_labels_"))
                metrics_file = str(evaluate.get("metrics_file", "metrics.paper.json"))
                metric_cmd = [py, str(self.root / "tools" / "evaluate_coral2025.py"),
                    "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"), "--data-type", str(task.dataset["data_type"]),
                    "--samples", str(samples), "--labels", str(labels), "--metrics-root", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats"),
                    "--output", str(run_dir / metrics_file)]
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
                phases.append(Phase(f"paper_metrics_w{omega}", metric_cmd, self.root, skip_if_exists=metric_outputs))

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
