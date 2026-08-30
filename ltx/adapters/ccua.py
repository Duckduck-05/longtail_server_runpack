from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import List

from .base import Adapter, Phase, resolve_inception_batch_size, resolve_num_workers
from ..config import Task


class CCUAAdapter(Adapter):
    """Contrastive Conditional-Unconditional Alignment (arXiv:2507.09052).

    Upstream ships two pipelines; this drives ``CCUA-DDPM``, the U-Net one, whose
    flag defaults already match the unified backbone (ch=128, [1,2,2,2], attn[1],
    2 blocks, EMA 0.9999, T=1000). Its long-tailed CIFAR split is the same
    ``ImbalanceCIFAR`` construction CBDM uses with ``rand_number=0``, so every
    row of the table trains on an identical subset.  For the deferred
    ImageNet-LT cell, ``data_type=imagenet_lt`` selects the bootstrap-installed
    manifest loader; this is important because the released ImageFolder path
    would otherwise train on all of ImageNet instead of the published LT split.

    The same adapter can express the plain DDPM objective by setting
    ``objective=ddpm`` (or explicit CCUA weights of zero) while retaining the
    CCUA U-Net/data/checkpoint plumbing.

    Sampling goes through ``main.py --sample`` rather than upstream
    ``evaluate.py``: the latter imports CCUA's own FLD/CLIP/DINOv2 metric stack
    at module scope, which this campaign replaces with one shared evaluator
    anyway, and which would otherwise pull a network-fetched dependency chain
    into every eval.
    """

    name = "ccua"

    @staticmethod
    def latest(run_dir: Path, target: int) -> int:
        values = []
        for p in run_dir.glob("ckpt_*.pt"):
            m = re.fullmatch(r"ckpt_(\d+)\.pt", p.name)
            if m and int(m.group(1)) < target:
                values.append(int(m.group(1)))
        return max(values, default=0)

    def _backbone_flags(self, train: dict) -> List[str]:
        """Pin the architecture explicitly instead of trusting flag defaults to
        keep agreeing with the other repos across vendored-source updates."""
        flags: List[str] = []
        if "ch" in train:
            flags.append(f"--ch={train['ch']}")
        flags += [f"--ch_mult={v}" for v in train.get("ch_mult", [])]
        flags += [f"--attn={v}" for v in train.get("attn", [])]
        for key in ("num_res_blocks", "ema_decay"):
            if key in train:
                flags.append(f"--{key}={train[key]}")
        return flags

    def phases(self, task: Task, batch_size: int | None = None) -> List[Phase]:
        repo = self.repo_dir(task)
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        py = task.runtime.get("python", "python")
        train, evaluate = task.train, task.eval
        inception_batch = resolve_inception_batch_size(evaluate)
        target = int(evaluate.get("checkpoint_step", train.get("total_steps", 300000)))
        batch = int(batch_size or train.get("batch_size", 64))
        num_class = task.dataset.get("num_class", task.dataset.get("num_classes", 100))
        img_size = task.dataset.get("img_size", 32)
        data_root = task.dataset.get("root", task.runtime.get("data_root", "./data"))

        method_cfg = task.method_config
        # Upstream's only U-Net-pipeline script sets both weights to 1.0; the
        # paper's alpha=gamma=0.05 is tuned for the DiT/SiT ImageNet pipeline,
        # whose latent-space loss scales are not comparable. Overridable per
        # method in the campaign YAML.
        objective = str(method_cfg.get("objective", task.method)).lower()
        if objective not in {"ddpm", "ccua"}:
            raise ValueError(f"CCUA U-Net adapter supports objective=ddpm or ccua, got {objective!r}")
        ccua_al = method_cfg.get("ccua_al", 0.0 if objective == "ddpm" else 1.0)
        ccua_ucl = method_cfg.get("ccua_ucl", 0.0 if objective == "ddpm" else 1.0)

        train_cmd = [
            py, "main.py", "--train",
            f"--data_type={task.dataset.get('data_type', 'cifar100lt')}",
            f"--data_path={data_root}",
            f"--imb_factor={task.dataset.get('imbalance_factor', 0.01)}",
            f"--num_class={num_class}", f"--img_size={img_size}",
            f"--logdir={run_dir}", f"--seed={task.seed}",
            f"--batch_size={batch}", f"--lr={train.get('lr', 2e-4)}",
            f"--T={train.get('T', 1000)}", f"--dropout={train.get('dropout', 0.1)}",
            f"--warmup={train.get('warmup', 5000)}",
            f"--save_step={train.get('save_step', 50000)}",
            f"--sample_step={train.get('sample_step', 100000)}",
            f"--num_workers={resolve_num_workers(train, 4)}",
            "--conditional", "--cfg",
            # CCUA is the alignment/contrastive pair alone. Naming the two
            # sibling objectives off explicitly means a changed upstream default
            # cannot silently turn this row into CBDM or T2H.
            "--nocbdm", "--notransfer_x0",
            f"--ccua_al={ccua_al}", f"--ccua_ucl={ccua_ucl}",
        ]
        if task.dataset.get("data_type") == "imagenet_lt":
            manifest = str(task.dataset.get("manifest", "")).strip()
            if not manifest:
                raise ValueError("ImageNet-LT requires dataset.manifest for the CCUA loader")
            train_cmd.append(f"--train_manifest={manifest}")
        train_cmd += self._backbone_flags(train)
        # The paper applies batch resampling only to ImageNet-LT and
        # TinyImageNet-LT, explicitly not to CIFAR-LT, so it stays off unless a
        # campaign asks for it.
        if method_cfg.get("brs", False):
            train_cmd += ["--brs", f"--brs_factor={method_cfg.get('brs_factor', 0.1)}"]
        train_cmd += list(map(str, method_cfg.get("flags", [])))

        # Upstream always iterates range(0, total_steps) and names checkpoints
        # step + ckpt_step, so a resume must be given the *remaining* budget or
        # it would train the full budget a second time.
        latest = self.latest(run_dir, target)
        remaining = (target - latest if latest else target) + 1
        train_cmd.append(f"--total_steps={remaining}")
        if latest:
            train_cmd += ["--resume", f"--resume_dir={run_dir}", f"--ckpt_step={latest}"]
        phases = [Phase("train", train_cmd, repo, skip_if_exists=[run_dir / f"ckpt_{target}.pt"])]

        # One trained checkpoint can be sampled at several guidance strengths:
        # omega only affects sampling. A single scale keeps the plain file names
        # so existing runs still resume; a sweep suffixes each artefact.
        scales = method_cfg.get(
            "guidance_scales",
            [method_cfg.get("guidance_scale", evaluate.get("guidance_scale", 1.0))],
        )
        if not isinstance(scales, list):
            scales = [scales]
        single = len(scales) == 1
        for omega in scales:
            suffix = "" if single else f"_w{omega}"
            samples = run_dir / f"ccua_samples{suffix}.npy"
            labels = Path(str(samples) + ".labels.npy")
            marker = run_dir / f"CCUA_EVAL_DONE{suffix}"
            eval_core = [
                py, "main.py", "--sample",
                f"--logdir={run_dir}", f"--seed={task.seed}",
                f"--ckpt_step={target}", f"--num_class={num_class}",
                f"--img_size={img_size}",
                f"--num_images={evaluate.get('num_images', 50000)}",
                f"--batch_size={evaluate.get('sample_batch_size', batch)}",
                f"--sample_method={evaluate.get('sample_method', 'ddpm')}",
                f"--ddim_skip_step={evaluate.get('ddim_skip_step', 1)}",
                f"--omega={omega}", f"--T={train.get('T', 1000)}",
                f"--dropout={train.get('dropout', 0.1)}",
                "--conditional", f"--sample_output={samples}",
            ]
            eval_core += self._backbone_flags(train)
            if evaluate.get("uniform_labels", False):
                eval_core.append("--uniform_labels")
            # The marker is written only after a full, successful sampling pass,
            # so a half-written array is never mistaken for a finished eval.
            quoted = " ".join(shlex.quote(str(x)) for x in eval_core)
            phases.append(Phase(
                f"eval{suffix}", ["bash", "-lc", f"{quoted} && touch {shlex.quote(str(marker))}"],
                repo, skip_if_exists=[marker],
            ))

            if not evaluate.get("paper_metrics", False):
                continue
            metrics_file = str(evaluate.get("metrics_file", "metrics.paper.json"))
            if not single:
                metrics_file = metrics_file.replace(".json", f"{suffix}.json")
            if task.dataset.get("data_type") == "imagenet_lt":
                reference_manifest = str(task.dataset.get("reference_manifest", "")).strip()
                if not reference_manifest:
                    raise ValueError("ImageNet-LT requires dataset.reference_manifest for metrics")
                metric_repo = Path(task.runtime["repos_root"]) / str(
                    task.method_config.get("metric_repo", "ImbDiff-CM")
                )
                phases.append(Phase(
                    f"paper_metrics{suffix}",
                    [
                        py, str(self.root / "tools" / "evaluate_imagenet_lt.py"),
                        "--repo", str(metric_repo),
                        "--image-root", str(data_root),
                        "--reference-manifest", reference_manifest,
                        "--samples", str(samples), "--labels", str(labels),
                        "--num-images", str(evaluate.get("num_images", 50000)),
                        "--image-size", str(img_size),
                        "--num-classes", str(num_class),
                        "--batch-size", str(inception_batch),
                        "--kid-repeats", str(evaluate.get("kid_repeats", 2)),
                        "--seed", str(task.seed),
                        "--output", str(run_dir / metrics_file),
                    ],
                    self.root,
                    skip_if_exists=[run_dir / metrics_file],
                ))
                continue
            metric_cmd = [
                py, str(self.root / "tools" / "evaluate_coral2025.py"),
                "--repo", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch"),
                "--data-type", str(task.dataset["data_type"]),
                "--samples", str(samples), "--labels", str(labels),
                "--metrics-root", str(Path(task.runtime["repos_root"]) / "CBDM-pytorch" / "stats"),
                "--output", str(run_dir / metrics_file),
                "--inception-batch-size", str(inception_batch),
            ]
            if evaluate.get("kid", False):
                metric_cmd += ["--kid", "--kid-subsets", str(evaluate.get("kid_subsets", 100)),
                               "--kid-subset-size", str(evaluate.get("kid_subset_size", 1000)),
                               "--kid-seed", str(evaluate.get("kid_seed", 2026))]
            per_class_file = str(evaluate.get("per_class_metrics_file", "")).strip()
            if per_class_file and single:
                metric_cmd += ["--per-class-output", str(run_dir / per_class_file),
                               "--longtail-groups", str(evaluate.get("longtail_groups", "none"))]
            outputs = [run_dir / metrics_file]
            if per_class_file and single:
                outputs.append(run_dir / per_class_file)
            phases.append(Phase(f"paper_metrics{suffix}", metric_cmd, self.root, skip_if_exists=outputs))
        return phases
