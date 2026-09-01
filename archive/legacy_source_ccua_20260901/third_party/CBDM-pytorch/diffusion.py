# from dd_code.backdoor.benchmarks.pytorch-ddpm.main import self
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from esc_objective import LEGACY_AMPLIFIED, low_t_multiplier

# The trainer is normally launched from third_party/CBDM-pytorch, while the
# checked projection solver is repository-root relative.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Training is launched from this third-party directory, which can itself carry
# a stale ``tools`` package.  Promote the repository root even when PYTHONPATH
# already listed it later, so the reviewed solver below has one authoritative
# import path in tests and production.
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))
from tools.barycentric_projection import (
    hard_calibrated_project_batch,
    simplex_project_batch,
)

import numpy as np
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def dist(X, Y):
    sx = torch.sum(X**2, dim=1, keepdim=True)
    sy = torch.sum(Y**2, dim=1, keepdim=True)
    return torch.sqrt(-2 * torch.mm(X, Y.T) + sx + sy.T)


def topk(y, all_y, K):
    dist_y = dist(y, all_y)
    return torch.topk(-dist_y, K, dim=1)[1]


def validate_training_protocol(protocol_mode, total_steps, save_step,
                               smoke_force_null_update, ckpt_step):
    """Validate fresh/full and resume-smoke update accounting."""
    if protocol_mode == 'full':
        if smoke_force_null_update:
            raise ValueError('smoke_force_null_update is forbidden in full mode')
        return
    if protocol_mode == 'smoke':
        if total_steps != 20 or save_step != 20 or smoke_force_null_update != 1:
            raise ValueError(
                'smoke requires exactly 20 steps/checkpoint and forced null update 1')
        return
    if protocol_mode == 'resume_smoke':
        if (
            ckpt_step <= 0
            or total_steps != ckpt_step + 20
            or save_step != 20
            or not ckpt_step < smoke_force_null_update <= total_steps
        ):
            raise ValueError(
                'resume_smoke requires total_steps=ckpt_step+20, save_step=20, '
                'and a forced null update in (ckpt_step,total_steps]')
        return
    raise ValueError('unknown protocol_mode')


def _closure_fixed_direction(batch_size, dimension, *, device, dtype, eps):
    """Fixed alternating-sign unit vector used by the generic norm control.

    Every sample receives (+1, -1, +1, ...) / sqrt(D); it is independent of
    model predictions and of the simplex residual.  The amplitude, but never
    the direction, is matched to CRT per sample.
    """
    indices = torch.arange(dimension, device=device)
    direction = torch.where(indices.remainder(2) == 0, 1., -1.).to(dtype)
    direction = direction / direction.norm().clamp_min(eps)
    return direction.unsqueeze(0).expand(batch_size, -1)


_CALIBRATED_TARGET_PRIOR_MODES = {'uniform', 'esc_effective'}


def _validated_probability_vector(values, *, num_class, name, require_positive=False):
    """Return a detached float64 probability vector with explicit invariants."""
    try:
        vector = torch.as_tensor(values, dtype=torch.float64).detach().clone()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability vector with {num_class} entries") from exc
    if vector.ndim != 1 or vector.numel() != num_class:
        raise ValueError(f"{name} must be a probability vector with {num_class} entries")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must contain only finite values")
    if require_positive:
        if not bool((vector > 0).all()):
            raise ValueError(f"{name} must contain only positive values")
    elif not bool((vector >= 0).all()):
        raise ValueError(f"{name} must be non-negative")
    if not torch.isclose(
            vector.sum(), torch.tensor(1.0, dtype=vector.dtype),
            rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return vector


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class, cfg, cb, tau, weight, finetune,
                 nbt_T_low=0, nbt_mix_lambda=1.0,
                 rgm_lambda=0.0, rgm_margin=0.0, rgm_t_low=0,
                 closure_mode='compute', closure_lambda=0.10,
                 closure_max_iter=60, closure_tol=1e-6, closure_eps=1e-12,
                 closure_calibrated_beta=0.0,
                 closure_calibrated_max_iter=4000,
                 closure_calibrated_tol=1e-8,
                 closure_enabled=True,
                 nbt_objective_mode=LEGACY_AMPLIFIED,
                 closure_calibrated_target_prior=None,
                 closure_calibrated_target_prior_mode='uniform',
                 closure_calibrated_sampling_prior=None):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg
        self.cb = cb
        self.tau = tau
        self.weight = weight
        self.finetune = finetune
        self.nbt_T_low = nbt_T_low
        self.nbt_mix_lambda = nbt_mix_lambda
        # This stays at the sampler because a mixed null branch must only
        # receive a low-t objective scalar on batches that actually used it.
        self.nbt_objective_mode = nbt_objective_mode
        low_t_multiplier(
            nbt_objective_mode, total_steps=self.T,
            low_steps=1 if nbt_T_low == 0 else nbt_T_low,
            used_low_t=False)
        self.rgm_lambda = rgm_lambda
        self.rgm_margin = rgm_margin
        self.rgm_t_low = rgm_t_low
        if closure_mode not in {'compute', 'norm', 'pairwise', 'crt'}:
            raise ValueError("closure_mode must be one of compute, norm, pairwise, crt")
        if closure_lambda < 0:
            raise ValueError("closure_lambda must be non-negative")
        if closure_max_iter <= 0 or closure_tol <= 0 or closure_eps <= 0:
            raise ValueError("Closure solver settings must be positive")
        if not 0.0 <= closure_calibrated_beta <= 1.0:
            raise ValueError("closure_calibrated_beta must be in [0, 1]")
        if closure_calibrated_max_iter <= 0 or closure_calibrated_tol <= 0:
            raise ValueError("calibrated field solver settings must be positive")
        if isinstance(num_class, bool) or int(num_class) != num_class or num_class <= 0:
            raise ValueError("num_class must be a positive integer")
        if closure_calibrated_target_prior_mode not in _CALIBRATED_TARGET_PRIOR_MODES:
            raise ValueError(
                "closure_calibrated_target_prior_mode must be one of uniform, esc_effective")
        if closure_calibrated_target_prior is None:
            target_prior = torch.full(
                (num_class,), 1.0 / num_class, dtype=torch.float64)
        else:
            target_prior = _validated_probability_vector(
                closure_calibrated_target_prior, num_class=num_class,
                name='closure_calibrated_target_prior')
        if closure_calibrated_target_prior_mode == 'esc_effective':
            if closure_calibrated_sampling_prior is None:
                raise ValueError(
                    "closure_calibrated_target_prior_mode=esc_effective requires "
                    "closure_calibrated_sampling_prior from the ESC objective contract")
            sampling_prior = _validated_probability_vector(
                closure_calibrated_sampling_prior, num_class=num_class,
                name='closure_calibrated_sampling_prior', require_positive=True)
        elif closure_calibrated_sampling_prior is not None:
            raise ValueError(
                "closure_calibrated_sampling_prior is only valid with "
                "closure_calibrated_target_prior_mode=esc_effective")
        else:
            sampling_prior = None
        self.closure_mode = closure_mode
        self.closure_enabled = closure_enabled
        self.closure_lambda = closure_lambda
        self.closure_max_iter = closure_max_iter
        self.closure_tol = closure_tol
        self.closure_eps = closure_eps
        self.closure_calibrated_beta = closure_calibrated_beta
        self.closure_calibrated_max_iter = closure_calibrated_max_iter
        self.closure_calibrated_tol = closure_calibrated_tol
        self.closure_calibrated_target_prior_mode = closure_calibrated_target_prior_mode
        self.register_buffer('closure_calibrated_target_prior', target_prior)
        self.register_buffer('closure_calibrated_sampling_prior', sampling_prior)
        # Updated on every forward so normal trainer logging can expose the
        # equal-compute protocol and numerical projection health.
        self.last_closure_defect = torch.tensor(0.)
        self.last_closure_target_grad_norm = torch.tensor(0.)
        self.last_closure_actual_grad_norm = torch.tensor(0.)
        self.last_closure_target_grad_norm_per_sample = torch.empty(0)
        self.last_closure_actual_grad_norm_per_sample = torch.empty(0)
        self.last_closure_aux_per_sample = torch.empty(0)
        self.last_closure_kkt_residual = torch.tensor(0.)
        self.last_closure_simplex_error = torch.tensor(0.)
        self.last_closure_active_size = torch.tensor(0.)
        self.last_closure_teacher_calls = 0
        self.last_closure_projection_calls = 0
        self.last_closure_target = None
        self.last_closure_calibrated_beta = torch.tensor(0.)
        self.last_closure_local_defect = torch.tensor(0.)
        self.last_closure_calibrated_defect = torch.tensor(0.)
        self.last_closure_calibrated_gap = torch.tensor(0.)
        self.last_closure_normalized_gap = torch.tensor(0.)
        self.last_closure_conditional_spread = torch.tensor(0.)
        self.last_closure_calibrated_constraint_error = torch.tensor(0.)
        self.last_closure_calibrated_fixed_residual = torch.tensor(0.)
        self.last_used_low_t = False
        self.last_nbt_multiplier = 1.0

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, y_0, augm=None, force_null_batch=False,
                closure_calibrated_beta=None):
        """
        Algorithm 1.
        """
        y_orig = y_0
        is_null_batch = False
        if force_null_batch and not (self.cfg or self.cb):
            raise ValueError('force_null_batch requires cfg or cb training')
        if (self.cfg or self.cb) and (force_null_batch or torch.rand(1)[0] < 1/10):
            y_0 = None
            is_null_batch = True

        # NB-LowT / ESC: concentrate null-branch timesteps in the identity-relevant
        # low-noise window. Defaults reproduce the original uniform timestep sampler.
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        used_low_t = False
        nbt_multiplier = 1.0
        if is_null_batch and self.nbt_T_low > 0:
            use_low_t = torch.rand(1, device=x_0.device)[0] < self.nbt_mix_lambda
            if use_low_t:
                t = torch.randint(self.nbt_T_low, size=(x_0.shape[0], ), device=x_0.device)
                used_low_t = True
                nbt_multiplier = low_t_multiplier(
                    self.nbt_objective_mode, total_steps=self.T,
                    low_steps=self.nbt_T_low, used_low_t=True)
        self.last_used_low_t = used_low_t
        self.last_nbt_multiplier = nbt_multiplier

        noise = torch.randn_like(x_0)

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        h = self.model(x_t, t, y=y_0, augm=augm)
        loss = F.mse_loss(h, noise, reduction='none')
        loss_reg = loss_com = torch.tensor(0).to(x_t.device)
        loss_rgm = torch.tensor(0).to(x_t.device)
        if self.cb and y_0 is not None:
            y_bal = torch.Tensor(np.random.choice(
                                 self.num_class, size=len(x_0),
                                 p=self.weight.numpy() if not self.finetune else None,
                                 )).to(x_t.device).long()

            h_bal = self.model(x_t, t, y=y_bal, augm=augm)
            weight = t[:, None, None, None] / self.T * self.tau
            loss_reg = weight * F.mse_loss(h, h_bal.detach(), reduction='none')
            loss_com = weight * F.mse_loss(h.detach(), h_bal, reduction='none')

        if is_null_batch and self.rgm_lambda > 0 and self.rgm_margin > 0:
            if self.rgm_t_low <= 0:
                rgm_mask = torch.ones_like(t, dtype=torch.bool)
            else:
                rgm_mask = t < self.rgm_t_low
            if rgm_mask.any():
                h_cond = self.model(x_t[rgm_mask], t[rgm_mask], y=y_orig[rgm_mask], augm=None).detach()
                residual = h[rgm_mask] - h_cond
                residual_rms = residual.float().pow(2).mean(dim=[1, 2, 3]).sqrt()
                loss_rgm = F.relu(self.rgm_margin - residual_rms).pow(2).mean() * self.rgm_lambda

        zero = torch.tensor(0., device=x_t.device)
        self.last_closure_defect = zero
        self.last_closure_target_grad_norm = zero
        self.last_closure_actual_grad_norm = zero
        self.last_closure_target_grad_norm_per_sample = torch.zeros(x_t.shape[0], device=x_t.device)
        self.last_closure_actual_grad_norm_per_sample = torch.zeros(x_t.shape[0], device=x_t.device)
        self.last_closure_aux_per_sample = torch.zeros(x_t.shape[0], device=x_t.device)
        self.last_closure_kkt_residual = zero
        self.last_closure_simplex_error = zero
        self.last_closure_active_size = zero
        self.last_closure_teacher_calls = 0
        self.last_closure_projection_calls = 0
        self.last_closure_target = None
        beta_now = (self.closure_calibrated_beta if closure_calibrated_beta is None
                    else float(closure_calibrated_beta))
        if not 0.0 <= beta_now <= 1.0:
            raise ValueError("closure_calibrated_beta must be in [0, 1]")
        self.last_closure_calibrated_beta = torch.tensor(beta_now, device=x_t.device)
        self.last_closure_local_defect = zero
        self.last_closure_calibrated_defect = zero
        self.last_closure_calibrated_gap = zero
        self.last_closure_normalized_gap = zero
        self.last_closure_conditional_spread = zero
        self.last_closure_calibrated_constraint_error = zero
        self.last_closure_calibrated_fixed_residual = zero
        if self.closure_enabled and is_null_batch and y_orig is not None:
            # Every arm executes the same teacher and projection workload. The
            # arm changes only the detached auxiliary target applied afterwards.
            # Teachers and projected targets are detached, so only the null
            # prediction receives an auxiliary gradient.
            B, D = h.shape[0], h[0].numel()
            with torch.no_grad():
                # ``no_grad`` does not disable dropout.  The barycentric target
                # must be built from one deterministic learned conditional
                # field, rather than from C independently dropped-out fields.
                # Keeping the teacher pass in eval mode also avoids advancing
                # the training RNG stream solely because Closure is enabled.
                was_training = self.model.training
                self.model.eval()
                try:
                    teachers = []
                    for cls in range(self.num_class):
                        labels = torch.full((B,), cls, device=x_t.device, dtype=torch.long)
                        teachers.append(self.model(x_t, t, y=labels, augm=augm).float())
                finally:
                    self.model.train(was_training)
                conditionals = torch.stack(teachers, dim=-1).reshape(B, D, self.num_class)
                projection = simplex_project_batch(
                    conditionals, h.detach().float().reshape(B, D),
                    max_iter=self.closure_max_iter, tol=self.closure_tol)
                local_target_flat = projection.projected
                conditional_spread = torch.zeros((), device=x_t.device, dtype=torch.float32)
                local_defect = None
                calibrated_defect = None
                calibrated_gap = None
                calibrated_constraint_error = 0.0
                calibrated_fixed_residual = 0.0
                target_flat = local_target_flat
                if beta_now > 0:
                    local_residual = h.detach().float().reshape(B, D) - local_target_flat
                    local_defect_per_state = local_residual.square().mean(dim=1)
                    conditional_spread_per_state = (
                        conditionals - conditionals.mean(dim=2, keepdim=True)
                    ).square().mean(dim=(1, 2))
                    target_prior = self.closure_calibrated_target_prior
                    sample_weights = None
                    if self.closure_calibrated_target_prior_mode == 'esc_effective':
                        # Null states are sampled from empirical rho, whereas
                        # the calibrated field is defined under the ESC
                        # effective law.  The solver normalizes this batch's
                        # target_prior[y_orig] / rho[y_orig] state weights.
                        sample_weights = (
                            target_prior[y_orig]
                            / self.closure_calibrated_sampling_prior[y_orig]
                        )
                        diagnostic_weights = sample_weights / sample_weights.sum()
                        local_defect = (
                            diagnostic_weights * local_defect_per_state.double()
                        ).sum().float()
                        conditional_spread = (
                            diagnostic_weights * conditional_spread_per_state.double()
                        ).sum().float()
                    else:
                        # Preserve the legacy reduction for the uniform arm.
                        local_defect = local_residual.square().mean()
                        conditional_spread = (
                            conditionals - conditionals.mean(dim=2, keepdim=True)
                        ).square().mean()
                    calibrated = hard_calibrated_project_batch(
                        conditionals, h.detach().float().reshape(B, D),
                        target_prior=target_prior,
                        sample_weights=sample_weights,
                        max_iter=self.closure_calibrated_max_iter,
                        tol=self.closure_calibrated_tol)
                    calibrated_target_flat = calibrated.projected
                    if sample_weights is None:
                        calibrated_defect = calibrated.residual.square().mean()
                    else:
                        calibrated_defect = (
                            diagnostic_weights
                            * calibrated.residual.square().mean(dim=1).double()
                        ).sum().float()
                    calibrated_gap = calibrated_defect - local_defect
                    target_flat = (
                        (1.0 - beta_now) * local_target_flat
                        + beta_now * calibrated_target_flat
                    )
                    calibrated_constraint_error = calibrated.constraint_error
                    calibrated_fixed_residual = calibrated.fixed_point_residual
                crt_target = target_flat.reshape_as(h).detach()
            self.last_closure_teacher_calls = self.num_class
            self.last_closure_projection_calls = 1 + int(beta_now > 0)
            crt_residual = (h.float().reshape(B, D) - crt_target.float().reshape(B, D))
            residual_norm = crt_residual.norm(dim=1)
            if local_defect is None:
                # This is the exact legacy reduction order at beta=0.
                local_defect = crt_residual.square().mean(dim=1).mean()
                calibrated_defect = local_defect
                calibrated_gap = torch.zeros_like(local_defect)
            # This is ||d [lambda * mean((h-p)^2)] / d h||_2 per sample.
            target_grad_norm = (2.0 * self.closure_lambda / D) * residual_norm
            # Preserve the legacy meaning of closure_defect_raw: D_local.  New
            # calibrated-target diagnostics are logged separately.
            self.last_closure_defect = local_defect.detach()
            self.last_closure_local_defect = local_defect.detach()
            self.last_closure_calibrated_defect = calibrated_defect.detach()
            self.last_closure_calibrated_gap = calibrated_gap.detach()
            self.last_closure_conditional_spread = conditional_spread.detach()
            self.last_closure_normalized_gap = (
                calibrated_gap / conditional_spread.clamp_min(self.closure_eps)
            ).detach()
            self.last_closure_calibrated_constraint_error = torch.tensor(
                calibrated_constraint_error, device=x_t.device).detach()
            self.last_closure_calibrated_fixed_residual = torch.tensor(
                calibrated_fixed_residual, device=x_t.device).detach()
            self.last_closure_target_grad_norm_per_sample = target_grad_norm.detach()
            self.last_closure_target_grad_norm = target_grad_norm.mean().detach()
            self.last_closure_kkt_residual = projection.kkt_residual.mean().detach()
            self.last_closure_simplex_error = projection.constraint_error.mean().detach()
            self.last_closure_active_size = projection.active_size.float().mean().detach()

            if self.closure_mode == 'crt':
                target = crt_target
            elif self.closure_mode == 'pairwise':
                # Keep the original-label conditional direction, rescaled so
                # lambda*MSE has exactly CRT's output-gradient norm per sample.
                with torch.no_grad():
                    original_target = torch.stack(teachers, dim=-1).gather(
                        -1, y_orig.view(B, *([1] * (h.ndim - 1)), 1).expand(*h.shape, 1)
                    ).squeeze(-1)
                    direction = h.detach().float().reshape(B, D) - original_target.float().reshape(B, D)
                    direction_norm = direction.norm(dim=1, keepdim=True)
                    normalized_direction = direction / direction_norm.clamp_min(self.closure_eps)
                    # A clamped near-zero original-label direction is not unit length,
                    # which silently makes the pairwise auxiliary gradient smaller than
                    # its CRT-matched target.  Use a deterministic direction that is
                    # independent of the projection residual in that degenerate case.
                    fallback = _closure_fixed_direction(
                        B, D, device=x_t.device, dtype=torch.float32, eps=self.closure_eps)
                    direction = torch.where(direction_norm > self.closure_eps,
                                            normalized_direction, fallback)
                    target = (h.detach().float().reshape(B, D) - residual_norm[:, None] * direction).reshape_as(h).detach()
            elif self.closure_mode == 'norm':
                with torch.no_grad():
                    direction = _closure_fixed_direction(B, D, device=x_t.device, dtype=torch.float32, eps=self.closure_eps)
                    target = (h.detach().float().reshape(B, D) - residual_norm[:, None] * direction).reshape_as(h).detach()
            else:  # compute: equal teacher/projection work but no auxiliary gradient.
                target = crt_target

            self.last_closure_target = target
            if self.closure_mode != 'compute' and self.closure_lambda > 0:
                auxiliary = F.mse_loss(h.float(), target.float(), reduction='none').mean(dim=[1, 2, 3])
                # The raw tensor is intentionally modified before NB-LowT's
                # nbt_multiplier below and before caller-side BNT weighting.
                loss = loss + self.closure_lambda * auxiliary[:, None, None, None]
                actual = (2.0 * self.closure_lambda / D) * (h.float().reshape(B, D) - target.float().reshape(B, D)).norm(dim=1)
                self.last_closure_aux_per_sample = auxiliary.detach()
                self.last_closure_actual_grad_norm_per_sample = actual.detach()
                self.last_closure_actual_grad_norm = actual.mean().detach()

        return loss * nbt_multiplier, loss_reg + 1/4 * loss_com, loss_rgm, is_null_batch

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_class, img_size=32, var_type='fixedlarge'):
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.num_class =    num_class
        self.img_size = img_size
        self.var_type = var_type
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y=None, omega=0.0, method='free'):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped}[self.var_type]

        model_log_var = extract(model_log_var, t, x_t.shape)
        unc_eps = None
        augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)

        # Mean parameterization
        eps = self.model(x_t, t, y=y, augm=augm)
        if omega > 0 and (method == 'cfg'):
            unc_eps = self.model(x_t, t, y=None, augm=None)
            guide = eps - unc_eps
            eps = eps + omega * guide
        
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, omega=0.0, method='cfg'):
        """
        Algorithm 2.
        """
        x_t = x_T.clone()
        y = None

        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0

                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y

    def _eps_with_cfg(self, x_t, t, y, omega, method):
        """CFG-blended eps prediction, factored out of p_mean_variance so the
        DDIM path (below) can reuse the exact same guidance semantics as the
        DDPM path above without touching it."""
        augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)
        eps = self.model(x_t, t, y=y, augm=augm)
        if omega > 0 and (method == 'cfg'):
            unc_eps = self.model(x_t, t, y=None, augm=None)
            eps = eps + omega * (eps - unc_eps)
        return eps

    def forward_ddim(self, x_T, omega=0.0, method='cfg', ddim_steps=50):
        """DDIM (Song et al. 2021) deterministic sampler, eta=0. Restored
        2026-07-02: the original ddim_steps flag/implementation used for the
        Jul-1 pilot early-evals (see outputs/*/res_ema_saved_N2048_STEP*.txt,
        which show 'DDIM: 50/50') had been lost from this file at some point
        during same-day inline patches (no DDIM code was present in the
        working tree or in `git diff` history as of 2026-07-02 09:xx UTC).
        Rewritten here to match the CFG omega-blending semantics of the
        existing DDPM `forward` above (reusing `_eps_with_cfg`), rather than
        ported verbatim from third_party/OC_LT/diffusion.py, whose sampler
        has a different (non-CFG) interface and is not a drop-in match.

        ddim_steps: number of sampling steps (e.g. 50 instead of T=1000).
        Does not modify `forward`/`p_mean_variance` (DDPM path unchanged).
        """
        x_t = x_T.clone()
        y = None
        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        # evenly spaced timestep subsequence, T-1 down to 0
        seq = torch.linspace(0, self.T - 1, steps=ddim_steps).long()
        seq = torch.unique(seq, sorted=True).flip(0).tolist()  # descending, e.g. [999,...,0]

        with torch.no_grad():
            for i, time_step in enumerate(tqdm(seq, total=len(seq), desc='DDIM')):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                eps = self._eps_with_cfg(x_t, t, y, omega, method)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                x_0 = torch.clip(x_0, -1., 1.)

                if i + 1 < len(seq):
                    t_next = seq[i + 1]
                    abar_next = self.alphas_bar[t_next]
                else:
                    abar_next = torch.tensor(1.0, device=x_t.device)  # t=-1 convention

                abar_next = abar_next.to(x_t.dtype)
                x_t = torch.sqrt(abar_next) * x_0 + torch.sqrt(1 - abar_next) * eps

        return torch.clip(x_t, -1, 1), y
