# Unified CIFAR-LT runpack

This private, standalone package produces one fair, report-ready baseline
table. It does not call another checkout and does not concatenate incompatible
paper tables.

On a CUDA server, run one command for the CIFAR-100-LT-only campaign (the
current scope — CIFAR-10-LT comes later):

```bash
bash scripts/run_server_c100.sh
```

It creates the pinned environment, loads the packaged `.env.local` W&B
settings, downloads CIFAR-100 through `torchvision`, prepares the shared
metric references, resumes safely when rerun, and launches all 27 CIFAR-100-LT
tasks (DDPM, CBDM, T2H, CM, CORAL, CCUA, IP-SVT and its two ablation arms
x seeds 0,1,2; `configs/unified_cifar_c100.yaml`).

### Current execution priority

For the first wave, run and validate only the three headline baselines across
all paired seeds: `ddpm`, `cbdm`, and `ccua`, each at seeds `0`, `1`, and `2`
(9 tasks total). This gives the baseline table and seed variance before any
GPU time is spent on T2H, CM, CORAL, or the IP-SVT/ablation rows. The remaining
18 tasks stay deferred until this first wave has completed training, sampling,
and metrics successfully.

This is an operational queue priority, not a change to the locked 27-task
fairness contract. Keep the same CIFAR-100-LT IF100 data, 300k-update budget,
50k class-uniform samples, DDIM-100 sampler, and metric protocol for all
methods when the deferred wave is started.

**DDPM evaluation cache.** The built-in Coral/DDPM evaluator must receive the
absolute balanced-reference file
`third_party/CBDM-pytorch/stats/cifar100.train.npz` (or its configured
`LTX_METRICS_ROOT` equivalent). It no longer guesses `./stats/...` from the
current working directory or silently selects another dataset's cache. The
canonical adapter passes this path automatically; after an old eval failure,
pull this runpack, rerun bootstrap/metric preparation, and rerun the failed
task. An existing final checkpoint is reused, so this does not retrain it.

**If you already have trained checkpoints, do not retrain.** Each task skips
its training phase when the final checkpoint is already in place, so dropping
checkpoints into the run directories turns the same command into an
evaluation-only pass. See
[Reusing checkpoints trained elsewhere](#reusing-checkpoints-trained-elsewhere).

GPU packing is automatic: the scheduler reads each GPU's free VRAM and packs
multiple tasks onto one GPU when there's room (starting from a 12 GB/task
estimate that self-corrects once a real task finishes), so the same command
adapts to whatever box it's rented on. Override it when needed:

```bash
bash scripts/run_server_c100.sh --per-gpu 3 --gpus 0,1,2,3   # pin the packing/GPU set
bash scripts/run_server_c100.sh --jobs 8                       # cap total concurrent tasks
```

or set `LTX_TASKS_PER_GPU` / `LTX_GPU_IDS` / `LTX_MAX_CONCURRENT` in
`.env.local`. Dataloader workers per task are divided automatically so packed
tasks don't oversubscribe the host's CPUs.

Once CIFAR-10-LT is back in scope, `bash scripts/run_server.sh` runs the full
54-task `configs/unified_cifar.yaml` campaign (all three cells) instead —
same machine, same environment, no extra setup.

## What runs

**Current scope — `scripts/run_server_c100.sh`:** 27 tasks, one cell.

| Data | Methods | Shared controls |
|---|---|---|
| CIFAR-100-LT IF100 | DDPM, CBDM, T2H, CM, CORAL, CCUA, IP-SVT, `ipsvt_twin`, `ipsvt_clean` | 300k updates; batch 64; LR 2e-4; U-Net ch=128 [1,2,2,2] attn[1] 2 blocks; EMA 0.9999; T=1000; 50k exact class-uniform samples; DDIM-100 at omega 1.5 |

### The three IP-SVT rows

IP-SVT adds a sparse, class-uniform auxiliary objective on top of the ordinary
DDPM loss:

```
L = L_DDPM^natural  +  lambda_aux ( L_twin + lambda_SVT * L_SVT )^class-uniform
```

`L_twin` asks a small neighbourhood of the class embedding to solve the same
exact DDPM task, which makes the denoising field locally robust to the class
condition. `L_SVT` matches the perturbed-condition response geometry to the
clean one (stopped), so that robustness is not bought by erasing the stochastic
variation the model already has.

| Row | Flags | Question it answers |
|---|---|---|
| `ipsvt` | `--ipsvt --ipsvt_mode=full --ipsvt_lambda_svt=1.0` | the method |
| `ipsvt_twin` | `--ipsvt --ipsvt_mode=twin --ipsvt_lambda_svt=0.0` | what does condition robustness alone buy? |
| `ipsvt_clean` | `--ipsvt --ipsvt_mode=clean` | is the gain just extra tail exposure? |

`ipsvt_clean` is the attribution control and the one most easily misread. It
runs a plain DDPM loss on the *same* class-uniform batches at the *same*
cadence with no condition perturbation at all. Class-uniform sampling by itself
gives tail classes more gradient updates, which would improve tail metrics for
reasons that have nothing to do with the mechanism; this row is what separates
the two. It matches the auxiliary branch's *data*, not its FLOPs — exposure is
the confound being removed, compute is not.

All three run the DDPM row's own trainer with extra flags. They are not a
fork: same data pipeline, same schedule, same sampler, same metric path, so a
difference between them and `ddpm` is a difference of objective and nothing
else. `--ipsvt` with `--amp` raises rather than silently skipping the auxiliary
branch, because a run labelled IP-SVT that trained the baseline objective is
worse than a crash.

`lambda_SVT` is frozen at 1. A pilot at 20k fine-tuning updates cleared all
four preregistered success criteria there, and at `lambda_SVT = 10` the same
statistic that improves tail coverage begins destroying class identity — a
dose-dependent trade-off, not a better setting waiting to be found. Sweeping it
inside the headline comparison would be hyperparameter selection on the test
table.

**Full protocol — `scripts/run_server.sh`:** 54 tasks, all three cells (the
same controls, plus CIFAR-10-LT IF100 and IF1000). This is the complete
locked comparison `configs/unified_cifar.yaml` defines; run it once CIFAR-10-LT
is back in scope.

`OC` is not an extra row: the official `OC_LT` repository calls its method
T2H. Running both names would double-count the same method.

**Why 300k updates.** Counted as images seen, 300k x 64 = 19.2M is exactly the
budget CBDM (300k x 64), CM (300k x 64) and CORAL (150k x 128) each used. A
smaller shared budget would run CBDM and CM below their own papers' design
point while running CORAL above its — undertraining two baselines is a
fairness violation in a way that a uniform surplus is not.

**Why DDIM-100 at omega 1.5.** Every row must reach the *same* sampler,
otherwise the table compares samplers as much as methods. Which sampler is a
protocol choice, and this campaign uses 100 DDIM steps.

That choice follows what the papers actually run. `OC_LT`'s published sampling
command passes `--w 1.5` and leaves `ddim_skip_step` at its default of 10
(= 100 steps); `CCUA` defaults the same way; `ImbDiff-CM`'s own
`configs/cifar100lt_ir100/cm.yaml` uses `omega: 1.5` with `ddim_skip_step: 20`
(= 50 steps). 100 is the majority default, and the extra cost over 50 is
immaterial next to training.

The campaign previously normalised onto the 1000-step ancestral chain, which is
twenty times what any of these papers run and cost ~14 h of sampling per task.
`coral-lt-diffusion` had no DDIM path at all, which is why that was the only
sampler every repository could reach; `patches/apply_coral_ddim.py` adds one,
copying CBDM's `forward_ddim` update rule so both repositories run the same
sampler rather than two implementations that share a name. Measured on this
box, 200 images: 410 s ancestral vs 75 s at DDIM-100.

The repositories spell the setting two ways — `cm`/`oc`/`ccua` take a skip
factor, the coral-family trainer takes a step count — so a config can satisfy
one and silently break the other. Preflight resolves both spellings against
`fairness_contract.sampler_family` and fails closed on a mismatch.

The source paper for each row is indexed in [papers/README.md](papers/README.md)
and fetched by `bash papers/fetch_papers.sh`.

Every completed row uses the same balanced CIFAR reference and reports FID,
KID, IS, F₈, F₁⁄₈, improved-PRD precision, and improved-PRD recall. It also
writes a separate per-class and Many/Medium/Few FID breakdown with its exact
sample counts. The resulting reports are written under
`runs/<campaign name>/report/` — `unified_cifar_c100_v1` for the current
CIFAR-100-LT-only run, `unified_cifar_v1` for the full campaign:

```text
runs/<campaign>/report/table.md
runs/<campaign>/report/tail_breakdown.md
runs/<campaign>/report/per_seed.csv
runs/<campaign>/report/tail_per_seed.csv
runs/<campaign>/report/summary.json
runs/<campaign>/report/results.log       # one self-contained log: fingerprint, vendor/env, per-task status, table, W&B links
runs/<campaign>/report/campaign_run.log  # snapshot of the whole-campaign stdout (bootstrap, GPU packing, launches, failures)
runs/<campaign>/latest.log               # live stdout of the most recent run (symlink, updates every launch)
```

The same per-seed and aggregate tables plus all task losses, samples, system
metrics, resolved commands, and artifacts are uploaded to the configured W&B
project. A missing seed or metric makes the report fail rather than silently
publishing a partial comparison.

The shared evaluator uses a separate memory-safe Inception micro-batch of 16
by default. This changes only how inference is split into GPU batches, not the
50k samples or any metric formula. Override `inception_batch_size` in the
campaign's `eval` block if the server has a different memory budget.

Given the `WANDB_API_KEY` in `.env.local`, the runner (via `--wandb`, which
both `run_server_c100.sh` and `run_server.sh` always pass) authenticates once
at bootstrap, flips the project to public-read, and publishes a W&B Report
combining the main table, the tail breakdown, and per-run comparison panels.
The report URL is printed at the end of the run and recorded in
`results.log` and `summary.json`.

Every file listed above is uploaded to that run as the `evaluation-report`
artifact, so someone running this on your behalf can hand over a W&B link
alone — the tables, the per-task status, and the full campaign stdout are all
readable there without shell access. Project visibility remains best-effort;
an online report-upload failure still preserves the local report but exits
non-zero, so it cannot be mistaken for a successful W&B hand-off.

## Reusing checkpoints trained elsewhere

Training is ~95% of a task's cost, so a checkpoint trained on another box
should never be retrained here. Every phase declares its outputs and is skipped
when they exist, and the train phase's output is the final checkpoint. A final
checkpoint in the expected run directory makes the same launch command run
evaluation only.

The path is derived from the campaign name, stage, method and seed:

```text
runs/unified_cifar_c100_v1/<stage>/<method>/seed_<n>/ckpt_300000.pt
```

```bash
# where every task expects its checkpoint, printed rather than guessed
python - <<'EOF'
from ltx.config import load_campaign
for t in sorted(load_campaign("configs/unified_cifar_c100.yaml").tasks,
                key=lambda x: (x.method, x.seed)):
    print(f"{t.method:12s} seed {t.seed}  {t.run_dir}/ckpt_{t.train['total_steps']}.pt")
EOF
```

Stages are `c100_if100_core` for `ddpm`/`cbdm`/`coral`/`ipsvt*`,
`c100_if100_t2h` for `t2h`, `c100_if100_cm` for `cm`, and `c100_if100_ccua`
for `ccua`.

For an intermediate checkpoint from another run, use an explicit resume
override. The path must be the checkpoint for the selected method/seed; a
single legacy `ema_model` checkpoint is a warm start, not an exact continuation:

```bash
python -m ltx.cli run --config configs/unified_cifar_c100.yaml \
  --resume-method ddpm --resume-seed 0 \
  --resume-checkpoint /path/to/ckpt_200000.pt \
  --resume-step 200000 --resume-mode ema_only
```

For per-seed files, use `{seed}` in the path and omit `--resume-seed`. A full
native checkpoint can use `--resume-mode full`; Coral then restores its model,
EMA, optimizer and scheduler state. The default is always fresh task
configuration plus automatic resume only from a checkpoint already inside
that task's own run directory. No checkpoint is inferred from another
campaign, and an EMA-only file is rejected unless `ema_only` is chosen
explicitly. The selected resume provenance is saved in `task.resolved.json`,
`RESUME_SOURCE.json` and W&B config.

Do not run a copied `/tmp/*.sh` or vendored `main.py` directly on the server:
that bypasses the scheduler/state database and the parent W&B run (the child
trainer is intentionally run with `WANDB_MODE=disabled`). Pull this repository
on the server and run `scripts/run_server_c100.sh`; with online mode it now
fails early if W&B credentials are missing instead of completing with no
online results.

Two things to check before trusting a transplanted checkpoint:

**It must be the same architecture and class count.** A CIFAR-10 checkpoint
loaded into a 100-class model fails with `size mismatch for
label_embedding.weight`, which is the good case — it fails loudly. A checkpoint
trained with a different `ch`/`ch_mult`/`num_res_blocks` may load and silently
evaluate a different model than the table claims, so confirm it was trained
under this protocol's backbone.

**It must have been trained by the same method.** Nothing in a `.pt` file
records which objective produced it. Dropping a DDPM checkpoint into
`ipsvt/seed_0/` produces a row labelled IP-SVT that is really DDPM, and no
check in this package will catch it. The `flagfile.txt` written next to each
checkpoint by its own training run is the only provenance there is — copy it
across with the checkpoint.

A task whose checkpoint is present still runs sampling and metrics, so the
campaign cost collapses to ~1.4 h per task instead of ~20 h.

## Operating a run in progress

Rerunning `bash scripts/run_server_c100.sh` is always the right command: it
re-bootstraps (which is idempotent), recovers tasks whose worker died, and
resumes each one from its newest checkpoint. Three things are worth knowing
before you stop or restart a campaign.

**Stopping.** `python -m ltx.cli stop --config configs/unified_cifar_c100.yaml`
SIGTERMs every running worker's process group. The next `run_server_c100.sh`
picks those tasks back up automatically (they are marked `retry`, not
`failed`) and resumes training from the last checkpoint. Checkpoints are
written every 50k updates and only the newest is kept, so a stop costs up to
50k updates (~5 h at 2.6 it/s) of whatever was training at the time. If you
expect to stop and restart repeatedly, lower `save_step` in
`configs/unified_cifar_c100.yaml` first — only one checkpoint is retained
either way, so it costs no extra disk.

**Failed tasks are not resumed.** A task that ended in `failed` stays failed;
`run` only picks up `pending`/`retry`. Requeue explicitly, then run again:

```bash
python -m ltx.cli retry-failed --config configs/unified_cifar_c100.yaml
bash scripts/run_server_c100.sh
```

A requeued task skips any phase whose outputs already exist, so a T2H task
that trained to `ckpt_300000.pt` and died in eval re-runs eval and metrics
only — it does not retrain.

**T2H/OC eval and `torch.compile`.** `OC_LT/main.py` wraps the U-Net in
`torch.compile`, so every key in the saved `net_model` carries an
`_orig_mod.` prefix, while `ddpm_gen.py` builds a plain `UNet`. Loading one
into the other raises `RuntimeError: Error(s) in loading state_dict for UNet`
and kills the eval phase *after* training has already been paid for.
`patches/apply_oc_compiled_ckpt.py` strips the prefix on the eval path, the
same way upstream already strips it in `ema()`; `scripts/bootstrap.sh` applies
it, so simply rerunning the launch script installs the fix. Preflight now
fails closed when the marker `third_party/OC_LT/.ltx_oc_compiled_ckpt_patch_v1`
is missing, rather than letting a campaign burn 300k updates into an eval that
cannot load its own checkpoint.

`configs/smoke_t2h.yaml` proves that path on a real GPU in about two minutes —
a 20-step train, then the production DDIM eval on a coarse stride:

```bash
source .venv/bin/activate
python -m ltx.cli run --config configs/smoke_t2h.yaml --skip-preflight
```

It writes `t2h_samples.npy`, its class-uniform labels, and a `SUCCESS` marker
under `runs/smoke_t2h_v1/`, and appears in W&B tagged `smoke`. FID/IS print as
`nan` there by design: 64 images is a plumbing check, not a measurement.

**Progress-bar log volume.** tqdm redraws with a carriage return, so with the
output piped every redraw used to become its own record: a 31-hour training
phase wrote ~300k of them into `stdout.log` (and the eval phase far more,
one 1,000-step bar per sampled batch), all of it also uploaded as a W&B
artifact. The worker now keeps one redraw per `progress_log_every_seconds`
(default 30) and sets `TQDM_MININTERVAL` (default 10) inside the child so the
redraws are thinned at the source too; both knobs live under `runtime:` in
`configs/server.yaml`, and `progress_log_every_seconds: 0` restores the old
behaviour. Lines that end in a real newline — ordinary logs, tracebacks,
metric prints, and the final state of each bar — are never dropped, and each
phase reports how many redraws it suppressed.

## Scope of the experimental programme

What this campaign is and is not, so the numbers are not asked to carry more
than they can.

**One cell, three seeds.** CIFAR-100-LT at IF100, seeds 0/1/2 for every row.
CIFAR-10-LT IF100 and IF1000 exist in `configs/unified_cifar.yaml` and are the
natural second cell; a second cell is what turns "IP-SVT helps here" into "IP-SVT
helps", so it is the first thing to add when compute allows.

**ImageNet-LT is out of scope.** `OC_LT` and `ImbDiff-CM` both report it and
`patches/cm_imagenet_lt.yaml` exists, but a 64x64 ImageNet-LT cell is far
beyond a single-GPU budget. CBDM and CORAL do not report it either, so its
absence is a limit on breadth rather than a hole in the comparison. **No paper
in this group evaluates generation on iNaturalist or Places-LT** — those are
long-tailed *classification* benchmarks, and adding them would not answer a
question the field is asking.

**Cost.** Measured on one H100: ~12 h (idle GPU) to ~36 h (contended) for 300k
updates, ~1.4 h to sample 50k images at DDIM-100, ~15 min for metrics. Call it
~20 h per task, so 27 tasks is roughly three weeks of exclusive GPU time. The
baselines are 18 of those 27 and are reusable across every future IP-SVT
change, which is why transplanting them rather than retraining is worth the
care documented above.

**Seeds are the cheapest thing to cut, and the wrong thing to cut first.** Most
papers in this area report a single seed. If the budget forces a reduction,
reduce the two ablation rows (`ipsvt_twin`, `ipsvt_clean`) to one seed before
touching the headline rows: those two answer a qualitative question — where does
the gain come from — while the main comparison is the quantitative claim and is
the one a reviewer will check for variance.

This is a new controlled benchmark, so it must be described as such in a
paper—not as a bit-for-bit reproduction of any individual paper table. Full
method and protocol audit: [UNIFIED_CIFAR_PROTOCOL.md](UNIFIED_CIFAR_PROTOCOL.md).
The CM/CORAL-derived experimental design, seed policy, paper-boundary audit,
and ablation/scaling plan are in
[EXPERIMENT_DESIGN_CM_CORAL.md](EXPERIMENT_DESIGN_CM_CORAL.md).
