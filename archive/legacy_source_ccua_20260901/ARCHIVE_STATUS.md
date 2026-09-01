# CCUA-only migration archive

Created 2026-09-01 after the active benchmark was consolidated onto the
native `third_party/CCUA-DDPM` U-Net.

The active runtime is:

- `third_party/CCUA-DDPM/` for DDPM, CBDM, CCUA, and IP-SVT;
- `third_party/CCUA-DDPM/stats/` for the shared CIFAR metric cache;
- `configs/native_cifar100_if100.yaml` and `scripts/run_server_c100.sh` for
  the current campaign.

The directories/files moved beside this file are historical source trees,
launch/config/test adapters, old protocol documents, and local run metadata.
They are retained here for audit and can be restored by moving them back to
their original paths. The only active helper retained outside this archive for
future CCUA ImageNet-LT work is `scripts/prepare_imagenet_lt.sh`.

The archived `runs/` entries contained only scheduler/task metadata; no
checkpoint, generated-sample, or metric payload was present. No `data/`,
active-run payload, checkpoint, generated sample, or metric-cache payload was
deleted.
