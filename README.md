# CCUA-DDPM unified long-tail runpack

## Canonical execution rule

All active CIFAR-100-LT experiments use one native host:
`third_party/CCUA-DDPM/` at commit
`baff9def6cfd553d4452d10eab0bac66ec2727aa`. DDPM, CBDM, T2H, CCUA, and
IP-SVT are objective choices dispatched through `ltx/adapters/ccua.py`; they
share the CCUA U-Net, dataset split, optimizer, checkpoint format, sampler,
and metric evaluator. Do not launch an active comparison from an archived
vendor tree.

The objective rows are:

| Row | CCUA-DDPM objective flags |
| --- | --- |
| DDPM | standard epsilon loss |
| CBDM | `--cbdm --cb_tau=1.0` |
| T2H | `--transfer_x0 --transfer_mode=t2h` |
| CCUA | native alignment and UCL losses |
| IP-SVT | native IP-SVT hook in `CCUA-DDPM/ipsvt_aux.py` |

T2H is a common-backbone port: its transfer objective is kept, while the old
OC/T2H source tree is archive-only. IP-SVT likewise changes only the objective
hook; it does not introduce a second U-Net host.

## Locked CIFAR-100-LT protocol

- CIFAR-100-LT IF100 (`imbalance_factor=0.01`), 32×32, 100 classes;
- 300,000 updates, batch size 64, learning rate `2e-4`, T=1000, EMA `0.9999`;
- 50,000 exact class-uniform generated labels;
- DDIM-100 (`ddim_skip_step=10`), guidance scale `1.5`;
- shared FID, KID, IS, F₈, F₁⁄₈, improved precision/recall, and
  Many/Medium/Few breakdowns.

The common metric/evaluation code is `tools/evaluate_ccua.py`, and the real
feature cache is `third_party/CCUA-DDPM/stats/`.

## Server commands

From the server checkout:

```bash
cd /home/nvidia-lab/ai4life/annd/respi/longtail_server_runpack
source .venv/bin/activate
export WANDB_MODE=offline                 # use online only with WANDB_API_KEY
export LTX_REPOS_ROOT="$PWD/third_party"
export LTX_DATA_ROOT="$PWD/data"
export LTX_RUNS_ROOT="$PWD/runs"
export LTX_METRICS_ROOT="$PWD/third_party/CCUA-DDPM/stats"
```

Validate a campaign before launching it:

```bash
python -m ltx.cli plan --config configs/native_cifar100_if100.yaml
python -m ltx.cli preflight --config configs/native_cifar100_if100.yaml
```

Launch the main 12-task campaign (DDPM, CBDM, CCUA, IP-SVT × seeds 0, 1, 2):

```bash
bash scripts/run_server_c100.sh
```

Launch the current seed-0 baseline/T2H campaign:

```bash
bash scripts/run_ccua_ddpm_cbdm.sh \
  configs/native_cifar100_if100_ddpm_cbdm_seed0.yaml
```

Launch only the six DDPM/CBDM baseline tasks:

```bash
bash scripts/run_ccua_ddpm_cbdm.sh \
  configs/native_cifar100_if100_ddpm_cbdm.yaml
```

The launcher bootstraps the pinned CCUA host, prepares the shared metric
assets, runs preflight, and schedules tasks. Set `LTX_RUNS_ROOT` explicitly on
each host so checkpoints and samples are written to the intended filesystem.
Never copy `.env.local` or credentials between hosts.

## Adding or changing a method

Keep the method in the CCUA host and add an explicit objective branch to
`ltx/adapters/ccua.py` plus a campaign config entry. Reuse the existing
dataset, U-Net, optimizer, checkpoint, sampling, and evaluator settings. A
new vendor checkout is not a valid comparison host.

Legacy source, old adapters, retired configs, and audit material are retained
under `archive/legacy_source_ccua_20260901/` for provenance only. They must not
be imported by active configs or bootstrap scripts.
