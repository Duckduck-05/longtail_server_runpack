"""
CVRD posterior-quality diagnostic (zero-training-of-the-diffusion-model check).

Implements Task C of Plan/IMPLEMENTATION_HANDOFF_2026-07-02.md (leakage-fixed CVRD spec).

LEAKAGE SAFETY:
  - q_hat_nu (the CLEAN posterior) is computed by a classifier head p_phi(c | x_t, t)
    trained on frozen EMA UNet features extracted from a NULL-conditioned forward pass
    (y=None). Its inputs are (x_t, t) only. It NEVER sees the true epsilon of the
    current sample and NEVER sees the true label as an input feature (only as the
    supervised target for the classifier's own cross-entropy training).
  - q_oracle additionally uses the true epsilon (||eps_EMA(x_t,c,t) - eps||^2). It is
    an upper-bound diagnostic ONLY. `--allow_oracle_in_loss` does not exist by design;
    q_oracle must never be wired into any training loss elsewhere in this repo.

The diffusion UNet (student and EMA) is frozen throughout this script. Only the small
p_phi classifier head is trained (diagnostic training, not CVRD/DRRD training).

Outputs (under --out_dir, default outputs/drrd_diagnostics/):
  - r2_summary.csv        one row per (alpha, tau_setting) with overall R2/acc/entropy
  - r2_by_tbin.csv         R2_clean / R2_oracle / acc / entropy per t-bin
  - r2_by_group.csv        R2_clean / R2_oracle / acc / entropy per class-group (head/mid/tail)
  - posterior_classifier_config.json   architecture + training config of p_phi
  - README.md              how to interpret, decision rule
  - p_phi_alpha{A}.pt       (optional, --save_classifier) trained classifier weights
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100  # noqa: E402
from model.model import UNet  # noqa: E402

# CIFAR-10 class order is alphabetical: airplane,automobile,bird,cat,deer,dog,
# frog,horse,ship,truck. With exp imbalance (img_max * imb_factor**(idx/9)),
# class 0 is head (n=5000) and class 9 is tail (n=50 @ imb_factor=0.01).
HEAD_CLASSES = [0, 1, 2]
MID_CLASSES = [3, 4, 5, 6]
TAIL_CLASSES = [7, 8, 9]
T_BIN_EDGES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def class_group(c: int) -> str:
    if c in HEAD_CLASSES:
        return "head"
    if c in TAIL_CLASSES:
        return "tail"
    return "mid"


def t_bin_label(t: int) -> str:
    for i in range(len(T_BIN_EDGES) - 1):
        if T_BIN_EDGES[i] <= t < T_BIN_EDGES[i + 1]:
            return f"[{T_BIN_EDGES[i]},{T_BIN_EDGES[i+1]})"
    return f"[{T_BIN_EDGES[-2]},{T_BIN_EDGES[-1]}]"


class PosteriorHead(nn.Module):
    """p_phi(c | x_t, t): small MLP over (GAP of frozen mid-block features, t-embedding).

    Inputs are exclusively x_t (noised image) and t (scalar timestep). No epsilon,
    no label, no conditional-branch output is used here. This is the leakage-safe path.
    """

    def __init__(self, feat_ch: int, num_class: int, hidden: int = 256, t_embed_dim: int = 64):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.net = nn.Sequential(
            nn.Linear(feat_ch + t_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_class),
        )

    def sinusoidal_t_embed(self, t: torch.Tensor, T: int = 1000) -> torch.Tensor:
        device = t.device
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float()[:, None] / T * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.t_embed_dim:
            emb = F.pad(emb, (0, self.t_embed_dim - emb.shape[-1]))
        return emb

    def forward(self, feat_gap: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.sinusoidal_t_embed(t)
        x = torch.cat([feat_gap, temb], dim=-1)
        return self.net(x)  # logits over classes


class MidBlockFeatureExtractor:
    """Non-invasive forward-hook feature extractor.

    We do NOT modify model/model.py. Instead we register a forward hook on the last
    middle-block module and capture its output. This is the "smallest safe
    alternative" mentioned in the task spec: it avoids editing shared model code
    (risky while other training jobs may be using the same file) and still gives
    genuine mid-block features (deepest, most compressed representation, right before
    the decoder starts using skip connections).
    """

    def __init__(self, unet: UNet):
        self.unet = unet
        self._feat = None
        self._hook = unet.middleblocks[-1].register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._feat = out

    def get_gap_features(self) -> torch.Tensor:
        assert self._feat is not None, "run a forward pass before reading features"
        feat = self._feat
        return feat.mean(dim=[2, 3])  # global average pool -> [B, C]

    def remove(self):
        self._hook.remove()


@torch.no_grad()
def batched_eps_all_classes(unet: UNet, x_t: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
    """Return eps_EMA(x_t, c, t) for every class c in [0,K). Shape [K, B, C, H, W]."""
    outs = []
    for c in range(K):
        y = torch.full((x_t.shape[0],), c, dtype=torch.long, device=x_t.device)
        outs.append(unet(x_t, t, y=y))
    return torch.stack(outs, dim=0)


def q_returns_from_class_counts(imb_factor: float, K: int = 10):
    img_max = 5000.0
    n_c = np.array([img_max * (imb_factor ** (c / (K - 1))) for c in range(K)])
    pi_c = n_c / n_c.sum()
    return n_c, pi_c


def build_nu(pi_c: np.ndarray, alpha: float) -> np.ndarray:
    nu_raw = pi_c ** (1.0 - alpha)
    return nu_raw / nu_raw.sum()


def forward_diffuse(x0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor):
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1).to(x0.dtype)
    sqrt_1mab = torch.sqrt(1 - alphas_bar[t]).view(-1, 1, 1, 1).to(x0.dtype)
    eps = torch.randn_like(x0)
    x_t = sqrt_ab * x0 + sqrt_1mab * eps
    return x_t, eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to ckpt_*.pt with 'ema_model' key")
    p.add_argument("--data_type", type=str, default="cifar10lt", choices=["cifar10lt", "cifar100lt"])
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--imb_factor", type=float, default=0.01)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--attn", type=int, nargs="+", default=[1])
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_class", type=int, default=10)
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0], help="nu tempering exponents to sweep")
    p.add_argument("--oracle_taus", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_val", type=int, default=512, help="validation subset size for R2/accuracy eval")
    p.add_argument("--n_classifier_train", type=int, default=2000, help="images used to train p_phi (subset of train set)")
    p.add_argument("--classifier_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32, help="kept small: K conditional forwards per batch item")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="outputs/drrd_diagnostics")
    p.add_argument("--save_classifier", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run: 16 train / 16 val images, 1 epoch, 2 t-bins worth of samples")
    args = p.parse_args()

    if args.smoke:
        args.n_val = 16
        args.n_classifier_train = 16
        args.classifier_epochs = 1
        args.batch_size = 8
        args.alphas = [0.5]
        args.oracle_taus = [1.0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([args.img_size, args.img_size]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize([args.img_size, args.img_size]),
    ])
    ds_cls = ImbalanceCIFAR10 if args.data_type == "cifar10lt" else ImbalanceCIFAR100
    train_dataset = ds_cls(root=args.root, imb_factor=args.imb_factor, train=True,
                           download=True, transform=train_transform)
    eval_dataset = ds_cls(root=args.root, imb_factor=args.imb_factor, train=True,
                          download=True, transform=eval_transform)
    K = args.num_class
    n_c, pi_c = q_returns_from_class_counts(args.imb_factor, K)
    print(f"[data] n_c={n_c.astype(int).tolist()}  pi_c={np.round(pi_c,4).tolist()}")

    # deterministic small subsets, disjoint (train subset for p_phi vs val subset for R2)
    all_idx = np.arange(len(train_dataset))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_idx)
    train_idx = all_idx[: args.n_classifier_train]
    val_idx = all_idx[args.n_classifier_train: args.n_classifier_train + args.n_val]
    if len(val_idx) == 0:
        val_idx = all_idx[: args.n_val]  # smoke fallback: allow overlap only in --smoke mode
    print(f"[data] p_phi train subset={len(train_idx)}  val subset={len(val_idx)} (disjoint={len(set(train_idx.tolist())&set(val_idx.tolist()))==0})")

    def make_loader(dataset, idx, batch_size, shuffle):
        subset = torch.utils.data.Subset(dataset, idx.tolist())
        return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=False)

    # ---- frozen EMA model ----
    unet = UNet(T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
                num_res_blocks=args.num_res_blocks, dropout=args.dropout,
                cond=True, augm=False, num_class=K).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["ema_model"] if "ema_model" in ckpt else ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    unet.load_state_dict(state)
    unet.eval()
    for p_ in unet.parameters():
        p_.requires_grad_(False)
    print(f"[model] loaded EMA weights from {args.ckpt}")

    betas = torch.linspace(args.beta_1, args.beta_T, args.T).double().to(device)

    extractor = MidBlockFeatureExtractor(unet)
    # probe feature channel count with a dummy null forward
    with torch.no_grad():
        dummy_x = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
        dummy_t = torch.zeros(2, dtype=torch.long, device=device)
        unet(dummy_x, dummy_t, y=None)
        feat_ch = extractor.get_gap_features().shape[-1]
    print(f"[model] mid-block feature channels = {feat_ch}")

    config_log = {
        "ckpt": args.ckpt,
        "feat_ch": feat_ch,
        "feature_source": "middleblocks[-1] output, global-average-pooled, NULL-conditioned forward (y=None)",
        "t_embed_dim": 64,
        "classifier_arch": "Linear(feat_ch+64,256)-ReLU-Linear(256,256)-ReLU-Linear(256,K)",
        "n_classifier_train": len(train_idx),
        "n_val": len(val_idx),
        "classifier_epochs": args.classifier_epochs,
        "leakage_note": "p_phi input is (x_t, t) only via NULL-conditioned mid-block features; "
                         "never sees true epsilon or true label as input (label is CE target only).",
        "oracle_note": "q_oracle uses true epsilon; upper-bound diagnostic ONLY; never used in any training loss.",
    }
    with open(out_dir / "posterior_classifier_config.json", "w") as f:
        json.dump(config_log, f, indent=2)

    # ---- train p_phi (diagnostic training only; diffusion model stays frozen) ----
    head = PosteriorHead(feat_ch, K).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    train_loader = make_loader(train_dataset, train_idx, args.batch_size, shuffle=True)

    print(f"[p_phi] training for {args.classifier_epochs} epoch(s) on {len(train_idx)} images...")
    for epoch in range(args.classifier_epochs):
        total_loss, n_seen = 0.0, 0
        for x0, y in train_loader:
            x0, y = x0.to(device), y.to(device)
            t = torch.randint(0, args.T, (x0.shape[0],), device=device)
            with torch.no_grad():
                x_t, _eps = forward_diffuse(x0, t, betas)
                unet(x_t, t, y=None)  # NULL-conditioned forward; populates hook
                feat = extractor.get_gap_features().float()
            logits = head(feat, t)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x0.shape[0]
            n_seen += x0.shape[0]
        print(f"  epoch {epoch}: CE loss = {total_loss / max(n_seen,1):.4f}")

    if args.save_classifier:
        torch.save(head.state_dict(), out_dir / "p_phi.pt")

    # ---- evaluation: R2_clean, R2_oracle, accuracy, entropy, by t-bin and by group ----
    head.eval()
    val_loader = make_loader(eval_dataset, val_idx, args.batch_size, shuffle=False)

    summary_rows = []
    tbin_records = []  # (alpha, tau_tag, t_bin, r2_clean_num, r2_clean_den, r2_oracle_num, r2_oracle_den, correct, total, entropy_sum, entropy_n)
    group_records = []  # same but keyed by class group

    for alpha in args.alphas:
        nu_c = build_nu(pi_c, alpha)
        log_nu = torch.tensor(np.log(nu_c + 1e-12), dtype=torch.float32, device=device)
        log_pi = torch.tensor(np.log(pi_c + 1e-12), dtype=torch.float32, device=device)

        for tau in args.oracle_taus:
            tag = f"alpha{alpha}_tau{tau}"
            acc_correct, acc_total = 0, 0
            ent_sum, ent_n = 0.0, 0
            r2c_num, r2c_den = 0.0, 0.0
            r2o_num, r2o_den = 0.0, 0.0
            per_tbin = {}
            per_group = {}

            with torch.no_grad():
                for x0, y in val_loader:
                    x0, y = x0.to(device), y.to(device)
                    t = torch.randint(0, args.T, (x0.shape[0],), device=device)
                    x_t, eps = forward_diffuse(x0, t, betas)

                    # clean posterior: p_phi from NULL-conditioned features (x_t,t only)
                    unet(x_t, t, y=None)
                    feat = extractor.get_gap_features().float()
                    logits = head(feat, t)  # [B,K], classifier logits ~ log p_pi(c|x_t)
                    log_p_pi = F.log_softmax(logits, dim=-1)
                    log_q_clean_unnorm = log_nu[None, :] - log_pi[None, :] + log_p_pi
                    q_clean = F.softmax(log_q_clean_unnorm, dim=-1)  # [B,K]

                    pred = log_p_pi.argmax(dim=-1)
                    acc_correct += (pred == y).sum().item()
                    acc_total += y.shape[0]
                    ent = -(q_clean * (q_clean + 1e-12).log()).sum(dim=-1)
                    ent_sum += ent.sum().item()
                    ent_n += y.shape[0]

                    # K conditional epsilon predictions from frozen EMA model
                    eps_all = batched_eps_all_classes(unet, x_t, t, K)  # [K,B,C,H,W]

                    eps_plug_clean = (q_clean.T[:, :, None, None, None] * eps_all).sum(dim=0)  # [B,C,H,W]

                    # oracle posterior (uses true eps) — diagnostic only
                    sq_err = ((eps_all - eps[None]) ** 2).mean(dim=[2, 3, 4])  # [K,B]
                    log_q_oracle_unnorm = log_nu[:, None] - sq_err / tau
                    q_oracle = F.softmax(log_q_oracle_unnorm, dim=0)  # [K,B]
                    eps_plug_oracle = (q_oracle[:, :, None, None, None] * eps_all).sum(dim=0)

                    num_clean = ((eps - eps_plug_clean) ** 2).mean(dim=[1, 2, 3])  # [B]
                    num_oracle = ((eps - eps_plug_oracle) ** 2).mean(dim=[1, 2, 3])
                    den = (eps ** 2).mean(dim=[1, 2, 3])

                    r2c_num += num_clean.sum().item()
                    r2c_den += den.sum().item()
                    r2o_num += num_oracle.sum().item()
                    r2o_den += den.sum().item()

                    t_np = t.cpu().numpy()
                    y_np = y.cpu().numpy()
                    for i in range(y.shape[0]):
                        tb = t_bin_label(int(t_np[i]))
                        gr = class_group(int(y_np[i]))
                        for store, key in [(per_tbin, tb), (per_group, gr)]:
                            if key not in store:
                                store[key] = dict(r2c_num=0.0, r2c_den=0.0, r2o_num=0.0, r2o_den=0.0,
                                                   correct=0, total=0, ent_sum=0.0)
                            store[key]["r2c_num"] += num_clean[i].item()
                            store[key]["r2c_den"] += den[i].item()
                            store[key]["r2o_num"] += num_oracle[i].item()
                            store[key]["r2o_den"] += den[i].item()
                            store[key]["correct"] += int(pred[i].item() == y[i].item())
                            store[key]["total"] += 1
                            store[key]["ent_sum"] += ent[i].item()

            r2_clean = 1.0 - r2c_num / max(r2c_den, 1e-12)
            r2_oracle = 1.0 - r2o_num / max(r2o_den, 1e-12)
            acc = acc_correct / max(acc_total, 1)
            mean_ent = ent_sum / max(ent_n, 1)
            print(f"[eval] {tag}: R2_clean={r2_clean:.4f} R2_oracle={r2_oracle:.4f} acc={acc:.4f} entropy={mean_ent:.4f}")
            summary_rows.append(dict(alpha=alpha, oracle_tau=tau, R2_clean=r2_clean, R2_oracle=r2_oracle,
                                      posterior_top1_acc=acc, posterior_entropy=mean_ent,
                                      n_val=acc_total))

            for tb, s in per_tbin.items():
                tbin_records.append(dict(
                    alpha=alpha, oracle_tau=tau, t_bin=tb,
                    R2_clean=1.0 - s["r2c_num"] / max(s["r2c_den"], 1e-12),
                    R2_oracle=1.0 - s["r2o_num"] / max(s["r2o_den"], 1e-12),
                    posterior_top1_acc=s["correct"] / max(s["total"], 1),
                    posterior_entropy=s["ent_sum"] / max(s["total"], 1),
                    n=s["total"],
                ))
            for gr, s in per_group.items():
                group_records.append(dict(
                    alpha=alpha, oracle_tau=tau, class_group=gr,
                    R2_clean=1.0 - s["r2c_num"] / max(s["r2c_den"], 1e-12),
                    R2_oracle=1.0 - s["r2o_num"] / max(s["r2o_den"], 1e-12),
                    posterior_top1_acc=s["correct"] / max(s["total"], 1),
                    posterior_entropy=s["ent_sum"] / max(s["total"], 1),
                    n=s["total"],
                ))

    extractor.remove()

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "r2_summary.csv", summary_rows)
    write_csv(out_dir / "r2_by_tbin.csv", tbin_records)
    write_csv(out_dir / "r2_by_group.csv", group_records)

    best = max(summary_rows, key=lambda r: r["R2_clean"]) if summary_rows else None
    decision = "UNKNOWN (no rows)"
    if best is not None:
        r2c, r2o = best["R2_clean"], best["R2_oracle"]
        if r2c >= 0.3:
            decision = f"GO: R2_clean={r2c:.3f} >= 0.3 -> CVRD/GRT can proceed with alpha={best['alpha']}."
        elif r2o >= 0.3:
            decision = (f"WEAK EXTRACTOR: R2_clean={r2c:.3f} < 0.15-0.3 but R2_oracle={r2o:.3f} >= 0.3 -> "
                        "information exists in conditional heads but p_phi/feature extraction is too weak. "
                        "Improve classifier (deeper head, more train steps, Tweedie re-noising ensemble) before CVRD.")
        else:
            decision = (f"NO-GO: R2_clean={r2c:.3f} and R2_oracle={r2o:.3f} both low -> "
                        "plug-in path is not worth training CVRD on. Fall back to IPW-renorm / TARL / GRT-with-uniform-prior-plug.")

    readme = f"""# CVRD posterior-quality diagnostic — results

Generated by `tools/cvrd_diagnostics.py`. Zero training of the diffusion model; only a
small classifier head p_phi(c|x_t,t) was trained (frozen UNet features).

## Leakage safety
- `R2_clean` uses the CLEAN posterior q_hat_nu(c|x_t,t) = (nu_c/pi_c) * p_phi(c|x_t,t).
  p_phi sees only (x_t, t) via a NULL-conditioned (y=None) forward pass through the
  frozen EMA UNet. It never receives the true epsilon of the sample as input.
- `R2_oracle` uses q_oracle(c|x_t,t) ~ nu_c * exp(-||eps_EMA(x_t,c,t)-eps||^2/tau),
  which DOES use the true epsilon. This is an UPPER BOUND diagnostic only and must
  never be used inside a training loss.

## Files
- `r2_summary.csv`: one row per (alpha, oracle_tau) sweep point, aggregated over the
  validation subset (n={args.n_val}).
- `r2_by_tbin.csv`: same metrics broken down by timestep bin {T_BIN_EDGES}.
- `r2_by_group.csv`: same metrics broken down by class group (head={HEAD_CLASSES},
  mid={MID_CLASSES}, tail={TAIL_CLASSES}).
- `posterior_classifier_config.json`: architecture/training config of p_phi.

## Decision rule (from Plan/IMPLEMENTATION_HANDOFF_2026-07-02.md S2)
- R2_clean >= 0.3 (check the t in [100,600] rows in r2_by_tbin.csv) -> CVRD/GRT can proceed.
- R2_clean < 0.15 but R2_oracle >= 0.3 -> posterior estimator (p_phi/feature extraction)
  is too weak; improve it (deeper classifier, more training, Tweedie re-noising ensemble)
  before committing GPU time to CVRD training.
- Both R2_clean and R2_oracle low -> the plug-in path is probably not worth training;
  fall back to IPW-renorm / TARL / GRT-with-uniform-prior-plug.

## This run's verdict
{decision}

## Caveats
- Validation subset is small (n={args.n_val}); treat R2 numbers as directional, not final,
  especially per-t-bin / per-group breakdowns which have even fewer samples each.
- p_phi was trained for only {args.classifier_epochs} epoch(s) on {len(train_idx)} images;
  this is a cheap diagnostic classifier, not a tuned one. A "WEAK EXTRACTOR" verdict
  should be re-checked after more p_phi training before concluding the plug-in path is dead.
- Mid-block features are captured via a forward hook (non-invasive); this is the
  "smallest safe alternative" — it avoids editing model/model.py while other jobs may
  be using that file, at the cost of only seeing one specific layer's representation.
"""
    with open(out_dir / "README.md", "w") as f:
        f.write(readme)

    print("\n" + decision)
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
