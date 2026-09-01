# Long-tail CIFAR-LT runpack

This private, standalone package produces one fair, report-ready baseline
table. The main table now uses the pinned CCUA-DDPM U-Net runner for all three
headline objectives (plain DDPM, CBDM, and CCUA), with the objective selected
explicitly in the released CCUA codepath. This is the codepath used by CCUA's
own CIFAR comparison and keeps the model, checkpoint schema, and official
DDIM sampler identical across the three rows. Coral/CBDM trees remain vendored
for audit and metric assets, but are not the main baseline runner. The
T2H-unified host is retained only for explicitly named secondary/legacy runs
and is not used to claim the main baseline result.

On a CUDA server, run one command for the CIFAR-100-LT-only campaign (the
current scope — CIFAR-10-LT comes later):

```bash
bash scripts/run_server_c100.sh
```

It creates the pinned environment, loads the packaged `.env.local` W&B
settings, downloads CIFAR-100 through `torchvision`, prepares the shared
metric references, and launches the nine native baseline tasks
(`configs/native_cifar100_if100.yaml`). Rerunning it resumes each native task
from its own CCUA-DDPM checkpoint and never transfers a DDPM checkpoint into
CBDM or CCUA.
On Viettel AI it automatically places new run state/checkpoints under
`/home/nvidia-lab/data_mount/longtail_server_runpack/runs` when no explicit
`LTX_RUNS_ROOT` is set; this avoids the full project-home filesystem.

### Current execution priority

Run and validate only the three headline baselines across all paired seeds:
`ddpm`, `cbdm`, and `ccua`, each at seeds `0`, `1`, and `2` (9 tasks total).
Temporarily pause CM, CORAL, T2H/unified, IP-SVT and its ablations. After the
nine baseline rows are complete, start the native IP-SVT experiment using the
same verified DDPM lineage; do not use a unified-host checkpoint as a native
warm start.

The outer native contract is fixed at CIFAR-100-LT IF100, 300k updates, batch
64, T=1000, 50k class-uniform samples, DDIM-100 at omega 1.5, and the shared
metric evaluator. Only the repository/objective implementation changes.
The CCUA paper's Table 7 reports a 200k-iteration reference, so this 300k
campaign is an explicitly labeled outer-control run; a paper-number claim
requires a separate 200k control with the same CCUA-DDPM sampler/evaluator.
See the [CCUA paper](https://arxiv.org/abs/2507.09052).

The ImageNet-LT experiment below is not part of that first wave. Before using
it as a comparison, all nine rows in `configs/native_cifar100_if100.yaml` must
have `SUCCESS` and a collected FID.

**Shared evaluation cache.** The native evaluator must receive the
absolute balanced-reference file
`third_party/CBDM-pytorch/stats/cifar100.train.npz` (or its configured
`LTX_METRICS_ROOT` equivalent). It never guesses `./stats/...` from the
current working directory or silently selects another dataset's cache. The
native adapters pass this path automatically; after an old eval failure, rerun
bootstrap/metric preparation and the failed task. An existing final
checkpoint is reused, so this does not retrain it.

**If you already have trained checkpoints, do not retrain.** Each task skips
its training phase when a valid namespaced checkpoint is already in place, so dropping
checkpoints into the run directories turns the same command into an
evaluation-only pass. See
[Reusing checkpoints trained elsewhere](#reusing-checkpoints-trained-elsewhere).

### Deferred ImageNet-LT setting (ACCESS or after the main table)

The requested secondary cell is isolated in
`configs/secondary_imagenet_lt.yaml`: ImageNet-LT at 64×64, 1,000 classes,
target training batch ≈256, DDPM and CCUA, seed 0, checkpoint/update 300k.
It reads the pinned ImageNet-LT train manifest rather than silently sampling a
new imbalance from full ImageNet, generates 50k exact class-uniform samples,
and reports ImageNet-LT FID/KID. The target batch is 256; the runner may retry
at a smaller batch after an OOM and records the effective batch in W&B.
This secondary config remains an explicitly isolated T2H/OC_LT host run for
now; it is not part of the CIFAR main claim and its DDPM/CCUA objective
dispatch is recorded separately in the run provenance.

Run this only on ACCESS (recommended because the licensed ImageNet payload is
large) or after the main CIFAR table is complete. The explicit gate prevents
accidental mixing with the main campaign:

```bash
# On ACCESS:
LTX_IMAGENET_LT_GATE=access bash scripts/run_imagenet_lt_secondary.sh

# After all 9 native CIFAR-100-LT baseline rows are complete:
LTX_IMAGENET_LT_GATE=main_complete bash scripts/run_imagenet_lt_secondary.sh
```

The launcher prepares and validates the ImageNet files/manifests, checks the
main-table gate, and writes the secondary report to
`runs/secondary_imagenet_lt_t2h_v1/report/`. Each method has its own
W&B run; the final comparison is also uploaded as a W&B table. Do not include
these two rows in the CIFAR main-table claim or run them as a replacement for
missing main-table rows.

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

The legacy full unified campaign can still be launched explicitly with
`bash scripts/run_unified_cifar.sh`, but `run_server.sh` now follows the native
baseline entrypoint and is not a shortcut to that archived 54-task table.

## Main native-repository policy

The main table keeps the objective implementations explicit while locking the
outer protocol: dataset, budget, backbone shape, seed list, label schedule,
sampler and evaluator. For this CCUA comparison, the released CCUA-DDPM U-Net
runner is the common baseline host; its `ddpm`, `cbdm`, and `ccua` branches
are separate objective settings, not a hidden unified research method.

| Method | Runtime source |
|---|---|
| DDPM | `third_party/CCUA-DDPM/` with the explicit DDPM objective |
| CBDM | `third_party/CCUA-DDPM/` with `--cbdm --cb_tau=1.0` |
| CCUA | `third_party/CCUA-DDPM/` with its native alignment/UCL losses |
| CM, CORAL, T2H/unified, IP-SVT | paused until the nine baseline rows finish |

The CCUA runner is an explicit objective dispatcher, not an always-on CCUA
loss. The three main rows resolve to `--nocbdm --ccua_al=0 --ccua_ucl=0`
(DDPM), `--cbdm --cb_tau=1.0 --ccua_al=0 --ccua_ucl=0` (CBDM), and
`--nocbdm --ccua_al=1 --ccua_ucl=1` (CCUA). All three then use the same
`main.py --sample --sample_method=ddim --ddim_skip_step=10` evaluator path.

The unified host and its checkpoints are not deleted while server jobs still
reference them, but they are removed from the main launcher and table. A
native checkpoint is never warm-started into another objective. The old server
DDPM checkpoint is a Coral-schema checkpoint with extra projection heads, so it
is retained as an audit artifact and deliberately not linked into the
CCUA-DDPM campaign. A cross-repository weight transplant would not be an exact
resume.

When changing a native method, update its adapter/config, run the native
preflight and focused tests, then launch through
`scripts/run_server_c100.sh`. Do not call a vendored `main.py` directly: that
bypasses the scheduler, checkpoint lineage and parent W&B run.

The minimum gate for a new campaign is:

```bash
python -m ltx.cli preflight --config configs/native_cifar100_if100.yaml
python -m pytest -q
```

## What runs

**Current scope — `scripts/run_server_c100.sh`:** 9 native baseline tasks, one cell.

| Data | Methods | Shared controls |
|---|---|---|
| CIFAR-100-LT IF100 | DDPM, CBDM, CCUA × seeds 0,1,2 | CCUA-DDPM runner; 300k updates; batch 64; LR 2e-4; U-Net ch=128 [1,2,2,2] attn[1] 2 blocks; EMA 0.9999; T=1000; 50k exact class-uniform samples; official DDIM-100 at omega 1.5 |

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

The IP-SVT rows are paused until the native baseline wave is complete. They
must then run from the native DDPM/Coral lineage with an explicitly documented
continuation or a fresh native run; a unified-host checkpoint cannot be used
as a silent warm start.

`lambda_SVT` is frozen at 1. A pilot at 20k fine-tuning updates cleared all
four preregistered success criteria there, and at `lambda_SVT = 10` the same
statistic that improves tail coverage begins destroying class identity — a
dose-dependent trade-off, not a better setting waiting to be found. Sweeping it
inside the headline comparison would be hyperparameter selection on the test
table.

The older full unified protocol remains available only as a legacy/secondary
experiment in `configs/unified_cifar.yaml`; it is not the current main table.

`OC` is not an extra row: the official `OC_LT` repository calls its method
T2H. Running both names would double-count the same method.

**Why 300k updates.** Counted as images seen, 300k x 64 = 19.2M is exactly the
budget CBDM (300k x 64), CM (300k x 64) and CORAL (150k x 128) each used. A
smaller shared budget would run CBDM and CM below their own papers' design
point while running CORAL above its — undertraining two baselines is a
fairness violation in a way that a uniform surplus is not.

**Why DDIM-100 at omega 1.5.** Every main-table row uses CCUA-DDPM's official
`sample_method=ddim` plus `ddim_skip_step=10` with T=1000. This fixes the
100-step grid and avoids comparing Coral's custom `[999,...,0]` schedule to
CCUA's official `[990,...,0]` schedule. Native preflight fails closed if this
sampler contract drifts.

The source paper for each row is indexed in [papers/README.md](papers/README.md)
and fetched by `bash papers/fetch_papers.sh`.

Every completed row uses the same balanced CIFAR reference and reports FID,
KID, IS, F₈, F₁⁄₈, improved-PRD precision, and improved-PRD recall. It also
writes a separate per-class and Many/Medium/Few FID breakdown with its exact
sample counts. The resulting reports are written under
`runs/<campaign name>/report/` — `native_cifar100_if100_v1` for the current
source-native CIFAR-100-LT baseline campaign:

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

For the native campaign, the report currently publishes `per_seed.csv`,
`summary.json` and `results.log`; the per-class aggregate remains in
`summary.json`. The report run also uploads these files and per-run metrics to
W&B so the remote operator does not need to copy logs manually.

The same per-seed and aggregate tables plus all task losses, samples, system
metrics, resolved commands, and artifacts are uploaded to the configured W&B
project. A missing seed or metric makes the report fail rather than silently
publishing a partial comparison.

The shared evaluator uses a separate memory-safe Inception micro-batch of 16
by default. This changes only how inference is split into GPU batches, not the
50k samples or any metric formula. Override `inception_batch_size` in the
campaign's `eval` block if the server has a different memory budget.

Given the `WANDB_API_KEY` in `.env.local`, the runner authenticates once at
bootstrap and each worker publishes its final native metrics to the configured
W&B project. The native report is best-effort W&B upload plus a fail-closed
local table; missing logging never turns a missing metric into a valid result.
The report URL is printed at the end of the run and recorded in
`results.log` and `summary.json`.

Every file listed above is uploaded to that run as the `evaluation-report`
artifact, so someone running this on your behalf can hand over a W&B link
alone — the tables, the per-task status, and the full campaign stdout are all
readable there without shell access. Project visibility remains best-effort;
an online report-upload failure still preserves the local report but exits
non-zero, so it cannot be mistaken for a successful W&B hand-off.

## Reusing checkpoints trained elsewhere

Training is ~95% of a task's cost, but a checkpoint is reusable here only when
its native provenance matches the selected method, seed, repository commit and
outer protocol. Every phase declares its outputs and is skipped when they
exist; otherwise the native adapter starts fresh or resumes from the latest
checkpoint in that same task directory.

The path is derived from the campaign name, stage, method and seed:

```text
runs/native_cifar100_if100_v1/<stage>/<method>/seed_<n>/ckpt_300000.pt
```

```bash
# where every task expects its checkpoint, printed rather than guessed
python - <<'EOF'
from ltx.config import load_campaign
for t in sorted(load_campaign("configs/native_cifar100_if100.yaml").tasks,
                key=lambda x: (x.method, x.seed)):
    print(f"{t.method:12s} seed {t.seed}  {t.run_dir}/ckpt_{t.train['total_steps']}.pt")
EOF
```

The native baseline stage is `c100_if100_core`; each method/seed has its own
subdirectory under that stage and uses the CCUA-DDPM adapter.

For an intermediate CCUA-DDPM checkpoint from another run, use an explicit
resume override only for the same method, seed, architecture and repository
commit. A
checkpoint from unified T2H, another objective, or another seed is not an
exact continuation and must not be supplied:

```bash
python -m ltx.cli run --config configs/native_cifar100_if100.yaml \
  --resume-method ddpm --resume-seed 0 \
  --resume-checkpoint /path/to/ckpt_200000.pt \
  --resume-step 200000 --resume-mode full
```

For per-seed files, use `{seed}` in the path and omit `--resume-seed`. The
default is always fresh task configuration plus automatic resume only from a
checkpoint already inside that task's own run directory. No checkpoint is
inferred from another campaign. The selected resume provenance is saved in
`task.resolved.json`, `RESUME_SOURCE.json` and W&B config where the native
trainer exposes it.

Do not run a copied `/tmp/*.sh` or vendored `main.py` directly on the server:
that bypasses the scheduler/state database and the parent W&B run (the child
trainer is intentionally run with `WANDB_MODE=disabled`). Pull this repository
on the server and run `scripts/run_server_c100.sh`; with online mode it now
fails early if W&B credentials are missing instead of completing with no
online results.

The old server result
`runs/unified_cifar_c100_v1/c100_if100_core/ddpm/seed_0/ckpt_300000.pt` remains
available for audit, but is not linked: its Coral projection-head schema is
not an exact CCUA-DDPM checkpoint. The new DDPM, CBDM, and CCUA rows therefore
start from their own CCUA-DDPM checkpoints.

Two things to check before trusting any other transplanted checkpoint:

**It must be the same architecture and class count.** A CIFAR-10 checkpoint
loaded into a 100-class model fails with `size mismatch for
label_embedding.weight`, which is the good case — it fails loudly. A checkpoint
trained with a different `ch`/`ch_mult`/`num_res_blocks` may load and silently
evaluate a different model than the table claims, so confirm it was trained
under this protocol's backbone.

**It must have been trained by the same method.** Native `ckpt_<step>.pt`
files are not self-authenticating across repositories. Confirm `task.json`,
`provenance.json`, repository commit, data split, model signature and seed
before linking one. A DDPM checkpoint cannot silently become a CBDM, CCUA or
IP-SVT row.

A task whose checkpoint is present still runs sampling and metrics, so the
campaign cost collapses to evaluation time instead of retraining.

## Operating a run in progress

Rerunning `bash scripts/run_server_c100.sh` is always the right command: it
re-bootstraps (which is idempotent), recovers tasks whose worker died, and
resumes each one from its latest native checkpoint. Three things are worth knowing
before you stop or restart a campaign.

**Stopping.** `python -m ltx.cli stop --config configs/native_cifar100_if100.yaml`
SIGTERMs every running worker's process group. The next `run_server_c100.sh`
picks those tasks back up automatically (they are marked `retry`, not
`failed`) and resumes training from the last native checkpoint. Checkpoints
are written every 50k updates, so a stop costs at most that resume interval.
If disk pressure matters, lower `save_step` in
`configs/native_cifar100_if100.yaml` before launching.

**Failed tasks are not resumed.** A task that ended in `failed` stays failed;
`run` only picks up `pending`/`retry`. Requeue explicitly, then run again:

```bash
python -m ltx.cli retry-failed --config configs/native_cifar100_if100.yaml
bash scripts/run_server_c100.sh
```

A requeued native task skips its training phase when `ckpt_300000.pt` exists,
then reruns only missing sampling/metrics phases. A failure in evaluation does
not retrain the completed model.

**Unified-host jobs.** Existing T2H-unified jobs are kept alive for provenance
and are not promoted into the main table. They can finish separately; their
checkpoints are never loaded by native baseline adapters.

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

**One cell, three seeds.** The current native main table is CIFAR-100-LT at
IF100, seeds 0/1/2 for DDPM/CBDM/CCUA. CIFAR-10-LT IF100 and IF1000 remain in
the archived unified protocol and can return only after the native main table
and native IP-SVT result are settled.

**ImageNet-LT is deferred, not part of the main table.** The optional
64×64 setting above runs only on ACCESS or after all 9 native CIFAR-100-LT main-table
rows are complete. It is a two-row baseline extension (DDPM/CCUA, seed 0),
not a substitute for missing CIFAR seeds and not evidence for a main-table
claim. **No paper in this group evaluates generation on iNaturalist or
Places-LT** — those are long-tailed *classification* benchmarks, and adding
them would not answer a question the field is asking.

**Cost.** Measured on one H100: ~12 h (idle GPU) to ~36 h (contended) for 300k
updates, ~1.4 h to sample 50k images at DDIM-100, ~15 min for metrics. Call it
~20 h per task. The nine baseline checkpoints are reusable across future
native IP-SVT changes, which is why transplanting them rather than retraining
is worth the care documented above.

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
