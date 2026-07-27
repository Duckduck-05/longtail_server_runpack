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

## Two separate paper reproductions

The command runs two **separate** suites and writes separate reports. They are
not a single joint leaderboard: identical method names across suites are not
interchangeable because the upstream implementation, training budget and metric
protocol differ. Do not compare or average rows across the two reports.

| Reproduction | Cells | Methods | Protocol |
| --- | --- | --- | --- |
| CM | CIFAR-10-LT IR100, CIFAR-100-LT IR100, ImageNet-LT 32/64 | DDPM, CBDM, OC, CM | CM source, 200k/300k steps, FID/KID |
| CORAL | CIFAR-10-LT IF100/IF1000, CIFAR-100-LT IF100 | DDPM, CBDM, T2H, CORAL | CORAL/OC sources, 150k/200k steps, FID/IS/F-scores/Recall |

Each has seeds `0,1,2`: 48 CM tasks and 36 CORAL tasks. CIFAR-10 IF100 and
CIFAR-100 IF100 occur in both paper protocols; DDPM/CBDM are intentionally
rerun there under each paper's own code and evaluation, not reused as the same
baseline result.

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
