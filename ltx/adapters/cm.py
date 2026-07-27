from __future__ import annotations

from pathlib import Path
from typing import List
import yaml

from .base import Adapter, Phase
from ..config import Task


class CMAdapter(Adapter):
    name = "cm"

    @staticmethod
    def latest_checkpoint(run_dir: Path, target: int) -> Path | None:
        best = None
        for p in run_dir.glob("ckpt_*.pt"):
            try: step = int(p.stem.split("_")[-1])
            except ValueError: continue
            if step < target and (best is None or step > int(best.stem.split("_")[-1])): best = p
        return best

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        py = task.runtime.get("python", "python")
        # The ImageNet-LT port uses the released CM architecture/loss together
        # with the generic LT_Dataset already present in its source tree.
        # CIFAR remains pinned to the authors' original config.
        image_net = task.dataset.get("data_type") == "imagenet_lt"
        if image_net:
            config_relative = "imagenet_lt/cm.yaml"
        elif task.dataset.get("data_type") == "cifar10lt":
            config_relative = "cifar10lt_ir100/cm.yaml"
        else:
            config_relative = "cifar100lt_ir100/cm.yaml"
        base_cfg = repo / "configs" / config_relative
        if not base_cfg.exists():
            raise FileNotFoundError(f"Official CM config missing: {base_cfg}")
        cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        cfg["method"] = task.method_config.get("upstream_method", cfg.get("method", "cm"))
        cfg["seed"] = int(task.seed)
        cfg["output_dir"] = str(run_dir)
        dataset_cfg = cfg.setdefault("dataset", {})
        dataset_cfg["root"] = task.dataset.get("root", task.runtime.get("data_root", "./data"))
        for key in ("name", "manifest", "num_classes", "img_size"):
            if key in task.dataset:
                dataset_cfg[key] = task.dataset[key]
        # All CIFAR implementations construct their LT split with the same
        # exponential imbalance formula and split seed.  The CM source names
        # the factor ``imb_factor`` while campaign configs use the clearer
        # ``imbalance_factor``.
        if "imbalance_factor" in task.dataset:
            dataset_cfg["imb_factor"] = float(task.dataset["imbalance_factor"])
        if "split_seed" in task.dataset:
            dataset_cfg["rand_number"] = int(task.dataset["split_seed"])
        for section, values in task.method_config.get("config_overrides", {}).items():
            cfg.setdefault(section, {}).update(values)
        total_steps = int(task.train.get("total_steps", cfg.get("training", {}).get("total_steps", 300001)))
        # The released CM loop is half-open.  Unified campaigns specify the
        # final update/checkpoint (e.g. 200000), so make that endpoint
        # inclusive without changing the historical paper-reproduction YAMLs.
        loop_bound = total_steps + int(bool(task.train.get("inclusive_final_step", False)))
        cfg.setdefault("training", {})["total_steps"] = loop_bound
        cfg["training"]["batch_size"] = int(batch_size or task.train.get("batch_size", cfg["training"].get("batch_size", 64)))
        for key in ("lr", "warmup", "T", "dropout", "num_workers", "ema_decay", "sample_step", "save_step"):
            if key not in task.train:
                continue
            target = "diffusion" if key == "T" else ("model" if key == "dropout" else "training")
            cfg.setdefault(target, {})[key if key != "T" else "T"] = task.train[key]
        checkpoint_step = int(task.eval.get("checkpoint_step", total_steps if task.train.get("inclusive_final_step", False) else total_steps - 1))
        image_dir = run_dir / f"generated-ckpt-{checkpoint_step}"
        feature_dir = run_dir / "features"
        prefix = f"CM-{task.stage}-{task.method}-s{task.seed}"
        cfg.setdefault("evaluation", {})["image_dir"] = str(image_dir)
        cfg["evaluation"]["feature_dir"] = str(feature_dir)
        cfg["evaluation"]["feature_prefix"] = prefix
        cfg["evaluation"]["num_images"] = int(task.eval.get("num_images", 50000))
        for key in ("guidance_scale", "sample_method", "ddim_skip_step", "sample_batch_size"):
            if key not in task.eval:
                continue
            cfg["evaluation"][{"guidance_scale": "omega", "sample_batch_size": "batch_size"}.get(key, key)] = task.eval[key]
        resolved = run_dir / "cm.resolved.yaml"
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        ckpt = run_dir / f"ckpt_{checkpoint_step}.pt"
        metrics = run_dir / "metrics.cm.json"
        sample_marker = run_dir / "CM_SAMPLE_DONE"
        import shlex
        sample_core = [py, "tools/sample_images.py", "--config", str(resolved), "--ckpt", str(ckpt)]
        unified_samples = run_dir / "samples.unified.npy"
        unified_labels = run_dir / "labels.unified.npy"
        if task.eval.get("metric_protocol") == "unified_cifar_v1":
            sample_core.extend(["--samples_output", str(unified_samples), "--labels_output", str(unified_labels)])
        sample_shell = " ".join(shlex.quote(str(x)) for x in sample_core) + " && touch " + shlex.quote(str(sample_marker))
        train_cmd = [py, "tools/train.py", "--config", str(resolved)]
        latest = self.latest_checkpoint(run_dir, checkpoint_step)
        if latest is not None:
            # Upstream ImbDiff-CM already implements checkpoint resume through
            # --ckpt_step.  Do not carry a brittle out-of-tree CLI patch.
            train_cmd += ["--ckpt_step", str(int(latest.stem.split("_")[-1]))]
        base_phases = [
            Phase("train", train_cmd, repo, skip_if_exists=[ckpt]),
            Phase("sample", ["bash", "-lc", sample_shell], repo,
                  skip_if_exists=[sample_marker, unified_samples, unified_labels] if task.eval.get("metric_protocol") == "unified_cifar_v1" else [sample_marker]),
        ]
        if image_net:
            reference_manifest = task.dataset.get("reference_manifest", "")
            if not reference_manifest:
                raise ValueError("ImageNet-LT requires dataset.reference_manifest")
            base_phases.append(Phase(
                "metrics_cm_imagenet_lt",
                [py, str(self.root / "tools" / "evaluate_cm_imagenet_lt.py"), "--repo", str(repo),
                 "--image-root", str(dataset_cfg["root"]), "--reference-manifest", str(reference_manifest),
                 "--generated-dir", str(image_dir), "--num-images", str(cfg["evaluation"]["num_images"]),
                 "--batch-size", str(cfg["evaluation"].get("batch_size", 128)), "--seed", str(task.seed),
                 "--output", str(metrics)],
                repo, skip_if_exists=[metrics],
            ))
            return base_phases
        # Unified Benchmark v1 deliberately uses the same array-based metric
        # implementation as DDPM/CBDM/T2H/CORAL.  CM emits PNGs, so first
        # canonicalize them to the exact [N, 3, 32, 32] / label contract.
        if task.eval.get("metric_protocol") == "unified_cifar_v1":
            metrics = run_dir / str(task.eval.get("metrics_file", "metrics.unified.json"))
            metric_cmd = [
                py, str(self.root / "tools" / "evaluate_coral2025.py"),
                "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"),
                "--data-type", str(task.dataset["data_type"]),
                "--samples", str(unified_samples), "--labels", str(unified_labels),
                "--metrics-root", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats"),
                "--output", str(metrics),
            ]
            if task.eval.get("kid", False):
                metric_cmd += ["--kid", "--kid-subsets", str(task.eval.get("kid_subsets", 100)),
                               "--kid-subset-size", str(task.eval.get("kid_subset_size", 1000)),
                               "--kid-seed", str(task.eval.get("kid_seed", 2026))]
            per_class_file = str(task.eval.get("per_class_metrics_file", "")).strip()
            if per_class_file:
                metric_cmd += ["--per-class-output", str(run_dir / per_class_file),
                               "--longtail-groups", str(task.eval.get("longtail_groups", "none"))]
            outputs = [metrics]
            if per_class_file:
                outputs.append(run_dir / per_class_file)
            return base_phases + [
                Phase(
                    "unified_metrics",
                    metric_cmd,
                    self.root, skip_if_exists=outputs,
                ),
            ]
        cifar = "cifar10" if task.dataset.get("data_type") == "cifar10lt" else "cifar100"
        return base_phases + [Phase(
            "metrics_cm_cifar_lt",
            [py, str(self.root / "tools" / "evaluate_cm_cifar_lt.py"), "--repo", str(repo), "--dataset", cifar,
             "--data-root", str(dataset_cfg["root"]), "--generated-dir", str(image_dir),
             "--num-images", str(cfg["evaluation"]["num_images"]),
             "--batch-size", str(cfg["evaluation"].get("batch_size", 128)), "--seed", str(task.seed),
             "--output", str(metrics)], repo, skip_if_exists=[metrics])]
