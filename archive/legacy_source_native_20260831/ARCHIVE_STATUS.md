# Legacy source-native snapshot

This archive contains the native method trees that the migrated comparison no
longer executes. The active benchmark uses `third_party/T2H-unified` for the
backbone, data loader, sampler, objective dispatch, and metrics.

Archived trees:

- `third_party/CBDM-pytorch`
- `third_party/CCUA-DDPM`
- `third_party/ImbDiff-CM`
- `third_party/OC_LT`
- `third_party/coral-lt-diffusion`

The snapshot intentionally excludes dataset payloads, metric caches,
generated outputs, and Python bytecode. On the server the live directories
remain in place until the legacy Coral/IP-SVT process releases them; only then
may the live trees be moved aside or replaced with archive links.

Migration host: `third_party/T2H-unified`.

`scripts/bootstrap.sh` now defaults to host mode and does not require or patch
these native trees. The old patch chain is retained only behind
`LTX_ENABLE_LEGACY_NATIVE=1`, which is set by the legacy reproduction
launchers.
