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
        for section, values in task.method_config.get("config_overrides", {}).items():
            cfg.setdefault(section, {}).update(values)
        total_steps = int(task.train.get("total_steps", cfg.get("training", {}).get("total_steps", 300001)))
        cfg.setdefault("training", {})["total_steps"] = total_steps
        cfg["training"]["batch_size"] = int(batch_size or task.train.get("batch_size", cfg["training"].get("batch_size", 64)))
        checkpoint_step = int(task.eval.get("checkpoint_step", total_steps - 1))
        image_dir = run_dir / f"generated-ckpt-{checkpoint_step}"
        feature_dir = run_dir / "features"
        prefix = f"CM-{task.stage}-{task.method}-s{task.seed}"
        cfg.setdefault("evaluation", {})["image_dir"] = str(image_dir)
        cfg["evaluation"]["feature_dir"] = str(feature_dir)
        cfg["evaluation"]["feature_prefix"] = prefix
        cfg["evaluation"]["num_images"] = int(task.eval.get("num_images", 50000))
        resolved = run_dir / "cm.resolved.yaml"
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        ckpt = run_dir / f"ckpt_{checkpoint_step}.pt"
        metrics = run_dir / "metrics.cm.json"
        sample_marker = run_dir / "CM_SAMPLE_DONE"
        import shlex
        sample_core = [py, "tools/sample_images.py", "--config", str(resolved), "--ckpt", str(ckpt)]
        sample_shell = " ".join(shlex.quote(str(x)) for x in sample_core) + " && touch " + shlex.quote(str(sample_marker))
        train_cmd = [py, "tools/train.py", "--config", str(resolved)]
        latest = self.latest_checkpoint(run_dir, checkpoint_step)
        if latest is not None:
            # Upstream ImbDiff-CM already implements checkpoint resume through
            # --ckpt_step.  Do not carry a brittle out-of-tree CLI patch.
            train_cmd += ["--ckpt_step", str(int(latest.stem.split("_")[-1]))]
        base_phases = [
            Phase("train", train_cmd, repo, skip_if_exists=[ckpt]),
            Phase("sample", ["bash", "-lc", sample_shell], repo, skip_if_exists=[sample_marker]),
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
        cifar = "cifar10" if task.dataset.get("data_type") == "cifar10lt" else "cifar100"
        return base_phases + [Phase(
            "metrics_cm_cifar_lt",
            [py, str(self.root / "tools" / "evaluate_cm_cifar_lt.py"), "--repo", str(repo), "--dataset", cifar,
             "--data-root", str(dataset_cfg["root"]), "--generated-dir", str(image_dir),
             "--num-images", str(cfg["evaluation"]["num_images"]),
             "--batch-size", str(cfg["evaluation"].get("batch_size", 128)), "--seed", str(task.seed),
             "--output", str(metrics)], repo, skip_if_exists=[metrics])]
