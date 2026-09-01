# Active vendor policy

`CCUA-DDPM/` is the only active model, data-loader, sampler, checkpoint, and
metric runtime for the current experiments. DDPM, CBDM, T2H, CCUA, and IP-SVT
are objective flags inside that same native CCUA U-Net host; they must not be
launched from separate vendor trees. T2H is dispatched through the CCUA host;
the archived OC/T2H checkout is not an active runtime.

The shared CIFAR metric cache lives at `CCUA-DDPM/stats/`. The small
`CCUA-DDPM/score/` namespace forwards to CCUA's released `score_new/`
implementation so the common evaluator does not require CBDM as a runtime
dependency.

Legacy Coral, CBDM, CM, OC/T2H, and IGD sources are historical audit material
only. They may be kept under `archive/` for reproducibility, but no active
configuration or bootstrap script should import them.

The pinned active revision and source provenance are recorded in
`THIRD_PARTY_MANIFEST.json`; local operational patch markers in
`CCUA-DDPM/.ltx_*` are required by preflight.
