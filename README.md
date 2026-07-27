# Long-tail diffusion baseline runpack

Standalone private hand-off for CM and CORAL long-tail diffusion baselines.
It includes vendored source, pinned environment, W&B configuration, automatic
dataset preparation and reproducible reporting.

## Run

The private `.env.local` is included for the intended hand-off. On a CUDA
server, run exactly:

```bash
bash scripts/run_server.sh
```

The command creates/updates the pinned CUDA environment, bootstraps the source
ports, downloads CIFAR-10/100 and ImageNet-LT, validates data/checksums, then
launches the full suites from scratch.

## What runs

- CM: DDPM, CBDM, OC, CM × seeds `0,1,2` on CIFAR-10-LT IR100,
  CIFAR-100-LT IR100, ImageNet-LT 32×32 and 64×64 (48 tasks).
- CORAL: DDPM, CBDM, T2H, CORAL × seeds `0,1,2` on its three published CIFAR
  cells (36 tasks).

CIFAR downloads through `torchvision`. ImageNet-LT downloads the original
ILSVRC2012 training archive, expands `train/<synset>/*`, and fetches
checksum-pinned long-tail manifests. See [data-source audit](IMAGE_NET_LT_SOURCE_AUDIT.md).

## Results

W&B receives loss, system state, samples, metrics and comparison tables.
Local outputs are written to:

```text
runs/cm_baselines_v1/report/table.md
runs/coral2025_cifar_v1/report/table.md
```

Reports are fail-closed: a missing seed or metric fails the comparison rather
than producing a partial table. This package is **baseline-only**; it does not
claim a proposed method wins until that method is implemented as a task.

## Notes

CM's ImageNet-LT configuration is a controlled source port because the public
CM release has no author-provided ImageNet-LT YAML. It is labelled as such in
the reports. For full operational detail, read [RUNBOOK_VI.md](RUNBOOK_VI.md).
