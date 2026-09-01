import copy
import hashlib
import json
import os
import random
import time
import warnings
from absl import app, flags
from tqdm import trange

import torch
from torch.amp import autocast
import numpy as np

from tensorboardX import SummaryWriter
import wandb
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from diffusion import (
    GaussianDiffusionSampler,
    GaussianDiffusionTrainer,
    validate_training_protocol,
)
from model.model import UNet
from utils.augmentation import *
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
from score.both import get_inception_and_fid_score
from utils.augmentation import KarrasAugmentationPipeline
from esc_objective import (
    LEGACY_AMPLIFIED,
    LEGACY_CLASS_MEAN,
    LOW_T_OBJECTIVE_MODES,
    WEIGHT_NORMALIZATIONS,
    build_esc_objective_contract,
    confusability_vector,
    runtime_null_class_weights,
    write_esc_objective_contract,
)


class _NullSummaryWriter:
    """No-op TensorBoard writer for disk-constrained resume runs."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_bool('improve', False, help='use improved diffusion network implemented by OpenAI')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_integer('seed', 41, help='global, CUDA, and DataLoader seed for paired runs')
flags.DEFINE_bool('deterministic', True, help='request deterministic Torch/CUDA algorithms')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('conditional', False, help='conditional generation')
flags.DEFINE_bool('weight', False, help='reweight')
flags.DEFINE_bool('cotrain', False, help='cotrain with an adjusted classifier or not')
flags.DEFINE_bool('logit', False, help='use logit adjustment or not')
flags.DEFINE_bool('augm', False, help='whether to use ADA augmentation')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with 10% probability')
# Dataset
flags.DEFINE_string('root', './', help='path of dataset')
flags.DEFINE_string('data_type', 'cifar100', help='data type, must be in [cifar10, cifar100, cifar10lt, cifar100lt]')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_float('num_class', 0, help='number of class of the pretrained model')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/', help='log directory')
flags.DEFINE_bool('disable_tensorboard', False,
                  help='disable TensorBoard event writes (useful for disk-constrained resumes)')
flags.DEFINE_integer('sample_size', 64, 'sampling size of images')
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_integer('private_num_images', 0, help='the number of private images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('sample_name', 'saved', help='name for a set of samples to be saved or to be evaluated')
flags.DEFINE_bool('sampled', False, help='evaluate sampled images')
flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [cfg, cond, uncond]')
flags.DEFINE_float('omega', 0.0, help='guidance strength for cfg sampling method')
flags.DEFINE_integer('ddim_steps', 0, help='restored 2026-07-02: if >0, use DDIM sampler with this many steps instead of full DDPM-T (does not change train-time behavior)')
flags.DEFINE_bool('prd', True, help='evaluate precision and recall (F_beta), only evaluated with 50k samples')
flags.DEFINE_bool('improved_prd', True, help='evaluate improved precision and recall, only evaluated with 50k samples')
# CBDM hyperparameters
flags.DEFINE_bool('cb', False, help='train with class-balancing(adjustment) loss')
flags.DEFINE_float('tau', 1.0, help='weight for the class-balancing(adjustment) loss')
# CBDM finetuning mechanism
flags.DEFINE_bool('finetune', False, help='finetuned based on a pretrained model')
flags.DEFINE_string('finetuned_logdir', '', help='logdir for the new model, where FLAGS.logdir will be the folder for \
                     the pretrained model')
flags.DEFINE_integer('ckpt_step', 0, help='step to reload the pretained checkpoint')
flags.DEFINE_bool(
    'exact_resume', True,
    help='require a native checkpoint with complete model, optimizer, RNG, and DataLoader state')
flags.DEFINE_bool(
    'allow_non_exact_resume', False,
    help='explicitly allow legacy EMA-only resume; it is not an exact continuation')
# CIDR: confusability-inverse DSM reweighting
flags.DEFINE_float('cidr_beta', 0.0, help='confusability multiplier (0=full_reweight, 1=CIDR)')
flags.DEFINE_float('cidr_alpha', 0.5, help='count exponent for CIDR weight')
# BNT: balanced null training (reweight only null-branch samples by inverse class freq)
flags.DEFINE_float('bnt_alpha', 0.0, help='null-branch reweight exponent (0=off, 0.5=BNT)')
# CANB: confusability-aware null-branch weighting, applied only together with BNT
flags.DEFINE_float('canb_beta', 0.0, help='null-branch confusability multiplier for CANB (0=off)')
flags.DEFINE_enum(
    'bnt_normalization', LEGACY_CLASS_MEAN, WEIGHT_NORMALIZATIONS,
    help='BNT/CANB normalization: legacy unweighted-class mean or empirical expectation')
# NB-LowT / ESC: null-branch low-timestep concentration
flags.DEFINE_integer('nbt_T_low', 0, help='upper timestep bound for null-branch low-t sampling (0=off)')
flags.DEFINE_float('nbt_mix_lambda', 1.0, help='probability of low-t sampling for null batches; <1 enables rho_mix')
flags.DEFINE_enum(
    'nbt_objective_mode', LEGACY_AMPLIFIED, LOW_T_OBJECTIVE_MODES,
    help='low-t null objective: historical amplification, low-window average, or truncated full objective')
# RGM: residual guidance margin on null branch, intended to raise class evidence/IS
flags.DEFINE_float('rgm_lambda', 0.0, help='residual guidance margin loss weight (0=off)')
flags.DEFINE_float('rgm_margin', 0.0, help='minimum RMS ||eps_null - stopgrad(eps_cond)|| for RGM')
flags.DEFINE_integer('rgm_t_low', 0, help='apply RGM only for t < rgm_t_low (0=all t)')
# CRT-C1: all modes retain the same conditional-teacher/projection compute.
flags.DEFINE_bool(
    'closure_enabled', True,
    help='enable Closure teacher/projection work; disable for matched baseline arms')
flags.DEFINE_enum('closure_mode', 'compute', ['compute', 'norm', 'pairwise', 'crt'], help='Closure arm; compute is equal-compute/no-auxiliary control')
flags.DEFINE_float('closure_lambda', 0.10, help='fixed CRT-C1 auxiliary coefficient')
flags.DEFINE_integer('closure_max_iter', 60, help='simplex projected-gradient iteration cap')
flags.DEFINE_float('closure_tol', 1e-6, help='simplex projected-gradient convergence tolerance')
flags.DEFINE_float('closure_eps', 1e-12, help='positive norm guard for Closure controls')
# Calibrated field target: opt-in interpolation between the legacy local hull
# projection and a hard batch-calibrated field projection.  Coefficients are
# solver variables only; no semantic posterior interpretation is assumed.
flags.DEFINE_float('closure_calibrated_beta', 0.0, help='final calibrated-target interpolation in [0,1]; every equal-compute Closure arm computes it when nonzero')
flags.DEFINE_integer('closure_calibrated_warmup', 0, help='new updates used to ramp calibrated beta from 0 to its final value')
flags.DEFINE_integer('closure_calibrated_max_iter', 4000, help='hard calibrated projected-gradient iteration cap')
flags.DEFINE_float('closure_calibrated_tol', 1e-8, help='hard calibrated fixed-point tolerance')
flags.DEFINE_enum(
    'closure_calibrated_target_prior_mode', 'uniform',
    ['uniform', 'esc_effective'],
    help='calibrated-field prior: legacy uniform or the active ESC effective prior')
# Evidence-only flags.  The driver supplies them only for the 20-update smoke.
flags.DEFINE_enum('protocol_mode', 'full', ['smoke', 'resume_smoke', 'full'], help='reviewed CRT protocol mode')
flags.DEFINE_integer('smoke_force_null_update', 0, help='smoke-only 1-indexed update forced onto CFG null branch')
flags.DEFINE_string('runtime_arm', '', help='CRT arm recorded in runtime evidence')
flags.DEFINE_string('source_manifest_sha256', '', help='frozen source manifest binding')
flags.DEFINE_string('protocol_sha256', '', help='frozen protocol binding')

device = torch.device('cuda:0')


_EXACT_RESUME_SCHEMA = 'cbdm-exact-resume-v1'
_EXACT_RESUME_REQUIRED_FIELDS = (
    'checkpoint_schema',
    'net_model',
    'ema_model',
    'optim',
    'sched',
    'fixed_x_T',
    'rng_state',
    'dataloader_state',
    'step',
)


def capture_runtime_rng_state():
    """Capture every RNG that can change a training update."""
    return {
        'torch_cpu_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_state_all': (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }


def restore_runtime_rng_state(state):
    """Restore the RNG state captured by :func:`capture_runtime_rng_state`."""
    required = ('torch_cpu_rng_state', 'torch_cuda_rng_state_all',
                'numpy_rng_state', 'python_rng_state')
    missing = [field for field in required if field not in state]
    if missing:
        raise RuntimeError(
            'checkpoint cannot satisfy --exact_resume; rng_state is missing: ' +
            ', '.join(missing))
    torch.set_rng_state(state['torch_cpu_rng_state'])
    cuda_states = state['torch_cuda_rng_state_all']
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; CUDA RNG state was saved but CUDA is unavailable')
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(state['numpy_rng_state'])
    random.setstate(state['python_rng_state'])


class DeterministicDataLoaderProgress:
    """Checkpointable cursor over a DataLoader with a dedicated generator.

    The cursor records the generator state at the start of the current epoch
    and the number of yielded batches.  Recreating that epoch and consuming
    the saved prefix reproduces the shuffled batch progression without saving
    dataset tensors or a worker process.
    """

    _VERSION = 1

    def __init__(self, dataloader, generator):
        self.dataloader = dataloader
        self.generator = generator
        self.epoch = 0
        self.batches_into_epoch = 0
        self._iterator = None
        self._epoch_start_generator_state = None

    def _start_epoch(self):
        self._epoch_start_generator_state = self.generator.get_state().clone()
        self._iterator = iter(self.dataloader)
        self.batches_into_epoch = 0

    def next_batch(self):
        while True:
            if self._iterator is None:
                self._start_epoch()
            try:
                batch = next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self._iterator = None
                continue
            self.batches_into_epoch += 1
            return batch

    def state_dict(self):
        if self._epoch_start_generator_state is None:
            # No batch has been drawn.  The next iterator must begin from the
            # generator state that was present at checkpoint time.
            epoch_start_state = self.generator.get_state().clone()
        else:
            epoch_start_state = self._epoch_start_generator_state.clone()
        return {
            'version': self._VERSION,
            'epoch': self.epoch,
            'batches_into_epoch': self.batches_into_epoch,
            'epoch_start_generator_state': epoch_start_state,
            'generator_state': self.generator.get_state().clone(),
            'loader_signature': {
                'dataset_length': len(self.dataloader.dataset),
                'batch_size': self.dataloader.batch_size,
                'drop_last': self.dataloader.drop_last,
                'num_workers': self.dataloader.num_workers,
            },
        }

    def load_state_dict(self, state):
        required = ('version', 'epoch', 'batches_into_epoch',
                    'epoch_start_generator_state', 'generator_state',
                    'loader_signature')
        missing = [field for field in required if field not in state]
        if missing:
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; dataloader_state is missing: ' +
                ', '.join(missing))
        if state['version'] != self._VERSION:
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; unsupported dataloader_state version')
        expected_signature = {
            'dataset_length': len(self.dataloader.dataset),
            'batch_size': self.dataloader.batch_size,
            'drop_last': self.dataloader.drop_last,
            'num_workers': self.dataloader.num_workers,
        }
        if state['loader_signature'] != expected_signature:
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; DataLoader configuration differs from checkpoint')
        batches_into_epoch = int(state['batches_into_epoch'])
        if batches_into_epoch < 0:
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; negative DataLoader batch cursor')

        epoch_start_state = state['epoch_start_generator_state']
        self.generator.set_state(epoch_start_state)
        self._epoch_start_generator_state = epoch_start_state.clone()
        self._iterator = iter(self.dataloader)
        self.epoch = int(state['epoch'])
        self.batches_into_epoch = 0
        try:
            for _ in range(batches_into_epoch):
                next(self._iterator)
                self.batches_into_epoch += 1
        except StopIteration as exc:
            raise RuntimeError(
                'checkpoint cannot satisfy --exact_resume; DataLoader cursor exceeds its epoch') from exc

        # RandomSampler consumes this dedicated generator only while an epoch
        # iterator is constructed.  Store and restore its post-construction
        # state exactly so the next epoch starts from the original permutation.
        self.generator.set_state(state['generator_state'])


def missing_exact_resume_fields(checkpoint):
    """Return the exact native-resume fields absent from ``checkpoint``."""
    missing = [field for field in _EXACT_RESUME_REQUIRED_FIELDS if field not in checkpoint]
    if checkpoint.get('checkpoint_schema') != _EXACT_RESUME_SCHEMA:
        if 'checkpoint_schema' not in missing:
            missing.append('checkpoint_schema (expected {})'.format(_EXACT_RESUME_SCHEMA))
    return missing


def require_exact_resume_checkpoint(checkpoint):
    """Reject historical or partial payloads before attempting a resume."""
    missing = missing_exact_resume_fields(checkpoint)
    if missing:
        raise RuntimeError(
            'checkpoint cannot satisfy --exact_resume; missing required fields: ' +
            ', '.join(missing))


def build_native_checkpoint_payload(*, net_model, ema_model, optim, sched,
                                    fixed_x_T, dataloader_progress, step):
    """Build the complete native payload required for an exact continuation."""
    return {
        'checkpoint_schema': _EXACT_RESUME_SCHEMA,
        'net_model': copy.deepcopy(net_model.state_dict()),
        'ema_model': copy.deepcopy(ema_model.state_dict()),
        'optim': copy.deepcopy(optim.state_dict()),
        'sched': copy.deepcopy(sched.state_dict()),
        'fixed_x_T': fixed_x_T.detach().cpu().clone(),
        'rng_state': capture_runtime_rng_state(),
        'dataloader_state': copy.deepcopy(dataloader_progress.state_dict()),
        'step': int(step),
    }


def restore_exact_checkpoint_payload(checkpoint, *, net_model, ema_model,
                                     optim, sched, dataloader_progress,
                                     fixed_x_T_device):
    """Restore all mutable state for an exact native-checkpoint continuation."""
    require_exact_resume_checkpoint(checkpoint)
    net_model.load_state_dict(checkpoint['net_model'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    optim.load_state_dict(checkpoint['optim'])
    sched.load_state_dict(checkpoint['sched'])
    dataloader_progress.load_state_dict(checkpoint['dataloader_state'])
    restored_fixed_x_T = checkpoint['fixed_x_T'].detach().clone().to(fixed_x_T_device)
    restore_runtime_rng_state(checkpoint['rng_state'])
    return restored_fixed_x_T


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        tgt_key = key.replace('_orig_mod.', '')
        target_dict[tgt_key].data.copy_(
            target_dict[tgt_key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def calibrated_beta_at_offset(final_beta, warmup_updates, update_offset):
    """Linear method warmup measured from the start of this train invocation."""
    if not 0.0 <= final_beta <= 1.0:
        raise ValueError('closure_calibrated_beta must be in [0, 1]')
    if warmup_updates < 0 or update_offset < 0:
        raise ValueError('calibrated warmup and update offset must be non-negative')
    if final_beta == 0.0:
        return 0.0
    if warmup_updates == 0:
        return float(final_beta)
    return float(final_beta) * min(float(update_offset) / float(warmup_updates), 1.0)


def resolve_closure_calibrated_target_prior(mode, *, num_class, esc_contract=None):
    """Select the calibrated-field prior and its state sampling law.

    ``esc_effective`` intentionally consumes the contract's serialized values
    rather than re-deriving either law from counts or runtime weights.
    """
    if mode == 'uniform':
        return torch.full((num_class,), 1.0 / num_class, dtype=torch.float64), None
    if mode != 'esc_effective':
        raise ValueError(
            'closure_calibrated_target_prior_mode must be uniform or esc_effective')
    if not isinstance(esc_contract, dict):
        raise ValueError(
            'closure_calibrated_target_prior_mode=esc_effective requires a '
            'compatible ESC objective contract')
    missing = [key for key in ('effective_prior', 'rho') if key not in esc_contract]
    if missing:
        raise ValueError(
            'closure_calibrated_target_prior_mode=esc_effective requires a '
            'compatible ESC objective contract containing {}'.format(', '.join(missing)))
    try:
        target_prior = torch.as_tensor(esc_contract['effective_prior'], dtype=torch.float64)
        sampling_prior = torch.as_tensor(esc_contract['rho'], dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'closure_calibrated_target_prior_mode=esc_effective requires a '
            'compatible ESC objective contract') from exc
    if target_prior.shape != (num_class,) or sampling_prior.shape != (num_class,):
        raise ValueError(
            'closure_calibrated_target_prior_mode=esc_effective requires ESC '
            'effective_prior and rho with {} entries'.format(num_class))
    return target_prior, sampling_prior


def calibrated_target_prior_hash(target_prior):
    """Return a stable, dtype-explicit identity for a calibrated target prior."""
    values = torch.as_tensor(target_prior, dtype=torch.float64).reshape(-1)
    payload = json.dumps(
        {'dtype': 'float64', 'values': [float(value).hex() for value in values]},
        sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('ascii')).hexdigest()


def closure_method_config(*, target_prior_mode, target_prior):
    """Serializable method identity persisted in every produced checkpoint."""
    target_prior_values = [float(value) for value in torch.as_tensor(
        target_prior, dtype=torch.float64).reshape(-1)]
    sampling_measure = (
        'importance_weighted_empirical_to_target'
        if target_prior_mode == 'esc_effective' else 'empirical_unweighted')
    if not FLAGS.closure_enabled:
        method_name = 'no_closure'
    elif FLAGS.closure_calibrated_beta > 0 and target_prior_mode == 'esc_effective':
        method_name = 'prior_coherent_closure'
    elif FLAGS.closure_calibrated_beta > 0:
        method_name = 'calibrated_field_target'
    else:
        method_name = 'closure'
    return {
        'name': method_name,
        'closure_enabled': FLAGS.closure_enabled,
        'closure_mode': FLAGS.closure_mode,
        'closure_lambda': FLAGS.closure_lambda,
        'closure_max_iter': FLAGS.closure_max_iter,
        'closure_tol': FLAGS.closure_tol,
        'closure_eps': FLAGS.closure_eps,
        'calibrated_beta_final': FLAGS.closure_calibrated_beta,
        'calibrated_warmup_updates': FLAGS.closure_calibrated_warmup,
        'calibrated_max_iter': FLAGS.closure_calibrated_max_iter,
        'calibrated_tol': FLAGS.closure_calibrated_tol,
        # Keep the original key for readers that only knew the uniform mode,
        # while making the exact prior/vector identity auditable.
        'calibrated_target_prior': target_prior_mode,
        'calibrated_target_prior_mode': target_prior_mode,
        'calibrated_target_prior_values': target_prior_values,
        'calibrated_target_prior_sha256': calibrated_target_prior_hash(target_prior),
        'sampling_measure': sampling_measure,
        'coefficient_semantics': 'solver_variables_only',
    }


def seed_worker(worker_id):
    # DataLoader derives this from its dedicated generator, not global order.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def set_global_seed():
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = FLAGS.deterministic
    if FLAGS.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def evaluate(sampler, model, sampled):
    if not sampled:
        model.eval()
        with torch.no_grad():
            images = []
            labels = []
            desc = 'generating images'
            for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
                batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
                x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
                if FLAGS.ddim_steps > 0:
                    batch_images, batch_labels = sampler.forward_ddim(x_T.to(device),
                                                         omega=FLAGS.omega,
                                                         method=FLAGS.sample_method,
                                                         ddim_steps=FLAGS.ddim_steps)
                else:
                    batch_images, batch_labels = sampler(x_T.to(device),
                                                         omega=FLAGS.omega,
                                                         method=FLAGS.sample_method)
                images.append((batch_images.cpu() + 1) / 2)
                if FLAGS.sample_method!='uncond' and batch_labels is not None:
                    labels.append(batch_labels.cpu())
            images = torch.cat(images, dim=0).numpy()
        np.save(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)), images)
        if FLAGS.sample_method != 'uncond':
            labels = torch.cat(labels, dim=0).numpy()
            np.save(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)), labels)
        model.train()
    else:
        labels = None
        images = np.load(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
                                            FLAGS.sample_method, FLAGS.omega,
                                            FLAGS.sample_name)))

        if FLAGS.sample_method != 'uncond':
            labels = np.load(os.path.join(FLAGS.logdir, '{}_{}_labels_ema_{}.npy'.format(
                                                FLAGS.sample_method, FLAGS.omega,
                                                FLAGS.sample_name)))
    save_image(
        torch.tensor(images[:256]),
        os.path.join(FLAGS.logdir, 'visual_ema_{}_{}_{}.png'.format(
                                    FLAGS.sample_method, FLAGS.omega, FLAGS.sample_name)),
        nrow=16)

    (IS, IS_std), FID, prd_score, ipr = get_inception_and_fid_score(
        images, labels, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, FLAGS=FLAGS)

    return (IS, IS_std), FID, prd_score, ipr


def train():
    validate_training_protocol(
        FLAGS.protocol_mode, FLAGS.total_steps, FLAGS.save_step,
        FLAGS.smoke_force_null_update, FLAGS.ckpt_step)
    if not FLAGS.runtime_arm or not FLAGS.source_manifest_sha256 or not FLAGS.protocol_sha256:
        raise ValueError('runtime evidence bindings are required')
    if not 0.0 <= FLAGS.closure_calibrated_beta <= 1.0:
        raise ValueError('closure_calibrated_beta must be in [0, 1]')
    if FLAGS.closure_calibrated_warmup < 0:
        raise ValueError('closure_calibrated_warmup must be non-negative')
    if FLAGS.closure_calibrated_max_iter <= 0 or FLAGS.closure_calibrated_tol <= 0:
        raise ValueError('calibrated field solver settings must be positive')
    if FLAGS.total_steps <= FLAGS.ckpt_step:
        raise ValueError('total_steps must exceed ckpt_step')
    if not 0.0 <= FLAGS.nbt_mix_lambda <= 1.0:
        raise ValueError('nbt_mix_lambda must lie in [0, 1]')

    resume_checkpoint = None
    resume_path = None
    resume_source = None
    if FLAGS.ckpt_step != 0:
        if FLAGS.allow_non_exact_resume:
            resume_mode = 'non_exact_legacy_ema'
        elif FLAGS.exact_resume:
            resume_mode = 'exact'
        else:
            raise ValueError(
                'checkpoint resume requires --exact_resume or --allow_non_exact_resume')
        resume_path = os.path.join(
            FLAGS.finetuned_logdir, 'ckpt_{}.pt'.format(FLAGS.ckpt_step))
        resume_checkpoint = torch.load(resume_path, map_location='cpu')
        if resume_mode == 'exact':
            require_exact_resume_checkpoint(resume_checkpoint)
            if int(resume_checkpoint['step']) != FLAGS.ckpt_step:
                raise RuntimeError(
                    'checkpoint cannot satisfy --exact_resume; checkpoint step {} does not match --ckpt_step {}'.format(
                        resume_checkpoint['step'], FLAGS.ckpt_step))
        resume_source = {
            'checkpoint': resume_path,
            'step': FLAGS.ckpt_step,
            'mode': resume_mode,
            'method_config': resume_checkpoint.get('method_config'),
        }

    set_global_seed()
    if FLAGS.augm:
        tran_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([FLAGS.img_size, FLAGS.img_size]),
            transforms.ToPILImage(),
            KarrasAugmentationPipeline(0.12),
        ])
    else:
        tran_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([FLAGS.img_size, FLAGS.img_size])
        ])


    if FLAGS.data_type == 'cifar10':
        dataset = CIFAR10(
                root=FLAGS.root,
                train=True,
                download=True,
                transform=tran_transform)
    elif FLAGS.data_type == 'cifar100':
        dataset = CIFAR100(
                root=FLAGS.root,
                # root='...',
                train=True,
                download=True,
                transform=tran_transform)
    elif FLAGS.data_type == 'cifar10lt':
        dataset = ImbalanceCIFAR10(
                root=FLAGS.root,
                # root='...',
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=tran_transform,
                target_transform=None,
                download=True)
    elif FLAGS.data_type == 'cifar100lt':
        dataset = ImbalanceCIFAR100(
                root='/GPFS/data/yimingqin/dd_code/backdoor/benchmarks/pytorch-ddpm/data',
                # root='...',
                imb_type='exp',
                imb_factor=FLAGS.imb_factor,
                rand_number=0,
                train=True,
                transform=tran_transform,
                target_transform=None,
                download=True)
    else:
        print('Please enter a data type included in [cifar10, cifar100, cifar10lt, cifar100lt]')

    data_generator = torch.Generator()
    data_generator.manual_seed(FLAGS.seed)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size,
        shuffle=True, num_workers=FLAGS.num_workers, drop_last=True,
        worker_init_fn=seed_worker, generator=data_generator)
    datalooper = DeterministicDataLoaderProgress(dataloader, data_generator)
    print('Dataset {} contains {} images with {} classes'.format(
        FLAGS.data_type, len(dataset.targets), len(np.unique(dataset.targets))))

    # get class weights for the current dataset
    def class_counter(all_labels):
        all_classes_count = torch.Tensor(np.unique(all_labels, return_counts=True)[1])
        return all_classes_count / all_classes_count.sum()
    weight = class_counter(dataset.targets)

    _n_c = torch.tensor(np.unique(dataset.targets, return_counts=True)[1], dtype=torch.float32)
    # C4 CIFAR-10 anchors are meaningful only for ten classes.  Other class
    # counts receive a neutral vector with the matching dimension.
    _confuse_c = torch.tensor(confusability_vector(len(_n_c)), dtype=_n_c.dtype)
    _n_max = _n_c.max()
    cidr_w = (1.0 + FLAGS.cidr_beta * _confuse_c) * (_n_max / _n_c) ** FLAGS.cidr_alpha
    cidr_w = cidr_w / cidr_w.mean()  # normalize so total loss scale is unchanged
    cidr_w = cidr_w.to(device)

    # BNT/CANB: balanced/confusability-aware null training, null-branch samples only.
    bnt_w_cpu = runtime_null_class_weights(
        _n_c, bnt_alpha=FLAGS.bnt_alpha, canb_beta=FLAGS.canb_beta,
        normalization=FLAGS.bnt_normalization)
    esc_contract = build_esc_objective_contract(
        _n_c.tolist(), bnt_alpha=FLAGS.bnt_alpha, canb_beta=FLAGS.canb_beta,
        normalization=FLAGS.bnt_normalization,
        low_t_objective_mode=FLAGS.nbt_objective_mode,
        total_steps=FLAGS.T, low_steps=FLAGS.nbt_T_low,
        low_t_enabled=FLAGS.nbt_T_low > 0 and FLAGS.nbt_mix_lambda > 0,
        low_t_mix_lambda=FLAGS.nbt_mix_lambda,
        cfg_enabled=FLAGS.cfg, cb_enabled=FLAGS.cb,
        null_reweight_enabled=FLAGS.bnt_alpha > 0,
        runtime_weights=bnt_w_cpu.tolist())
    bnt_w = bnt_w_cpu.to(device)

    # model setup
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
    calibrated_target_prior, calibrated_sampling_prior = (
        resolve_closure_calibrated_target_prior(
            FLAGS.closure_calibrated_target_prior_mode,
            num_class=FLAGS.num_class, esc_contract=esc_contract))
    method_config = closure_method_config(
        target_prior_mode=FLAGS.closure_calibrated_target_prior_mode,
        target_prior=calibrated_target_prior)
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)
    if resume_checkpoint is not None and FLAGS.allow_non_exact_resume:
        net_model.load_state_dict(resume_checkpoint['ema_model'])
    ema_model = copy.deepcopy(net_model)
    net_model = torch.compile(net_model)

    # training setup
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, dataset,
        FLAGS.num_class, FLAGS.cfg, FLAGS.cb, FLAGS.tau, weight, FLAGS.finetune,
        FLAGS.nbt_T_low, FLAGS.nbt_mix_lambda,
        FLAGS.rgm_lambda, FLAGS.rgm_margin, FLAGS.rgm_t_low,
        closure_mode=FLAGS.closure_mode, closure_lambda=FLAGS.closure_lambda,
        closure_max_iter=FLAGS.closure_max_iter, closure_tol=FLAGS.closure_tol,
        closure_eps=FLAGS.closure_eps,
        closure_enabled=FLAGS.closure_enabled,
        closure_calibrated_beta=FLAGS.closure_calibrated_beta,
        closure_calibrated_max_iter=FLAGS.closure_calibrated_max_iter,
        closure_calibrated_tol=FLAGS.closure_calibrated_tol,
        nbt_objective_mode=FLAGS.nbt_objective_mode,
        closure_calibrated_target_prior=calibrated_target_prior,
        closure_calibrated_target_prior_mode=FLAGS.closure_calibrated_target_prior_mode,
        closure_calibrated_sampling_prior=calibrated_sampling_prior).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    else:
        print('LOGDIR already exists.')
    write_esc_objective_contract(
        os.path.join(FLAGS.logdir, 'ESC_OBJECTIVE_CONTRACT.json'), esc_contract)
    writer = _NullSummaryWriter() if FLAGS.disable_tensorboard else SummaryWriter(FLAGS.logdir)
    writer.add_text('method_config', json.dumps(method_config, sort_keys=True), FLAGS.ckpt_step)
    writer.flush()
    wandb.init(
        project="longtail-baselines", name="cbdm",
        config={**FLAGS.flag_values_dict(), **method_config}, resume="allow")
    
    # Restore after all construction/logging side effects so the next update
    # observes precisely the RNG state captured after the saved update.
    if resume_checkpoint is not None and not FLAGS.allow_non_exact_resume:
        fixed_x_T = restore_exact_checkpoint_payload(
            resume_checkpoint, net_model=net_model, ema_model=ema_model,
            optim=optim, sched=sched, dataloader_progress=datalooper,
            fixed_x_T_device=device)
    else:
        # Fix generation noise for comparable fresh and explicitly non-exact runs.
        fixed_x_T = torch.randn(
            min(FLAGS.sample_size, 100), 3, FLAGS.img_size, FLAGS.img_size).to(device)

    # Backup all arguments and record the exact paired initialization before
    # any optimizer update.  The hash includes names, dtype, shape, and bytes.
    with open(os.path.join(FLAGS.logdir, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())
    initial_state_hasher = hashlib.sha256()
    for name, value in net_model.state_dict().items():
        value_cpu = value.detach().cpu().contiguous()
        initial_state_hasher.update(name.encode('utf-8'))
        initial_state_hasher.update(str(value_cpu.dtype).encode('ascii'))
        initial_state_hasher.update(str(tuple(value_cpu.shape)).encode('ascii'))
        initial_state_hasher.update(value_cpu.numpy().tobytes())
    initial_model_state_sha256 = initial_state_hasher.hexdigest()
    with open(os.path.join(FLAGS.logdir, 'INITIAL_MODEL_STATE.json'), 'w') as f:
        json.dump({
            'sha256': initial_model_state_sha256,
            'seed': FLAGS.seed,
            'checkpoint_resume_step': FLAGS.ckpt_step,
            'fresh_initialization': FLAGS.ckpt_step == 0,
        }, f, indent=2, sort_keys=True)

    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.monotonic()
    closure_batches = []

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.ckpt_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, y_0 = datalooper.next_batch()

            # when using ADA, the augmentation parameters will also be returned by the dataloader
            augm = None
            if type(x_0) == list:
                x_0, augm = x_0
                augm = augm.to(device)

            x_0 = x_0.to(device)
            y_0 = y_0.to(device)

            calibrated_beta_now = calibrated_beta_at_offset(
                FLAGS.closure_calibrated_beta,
                FLAGS.closure_calibrated_warmup,
                step - FLAGS.ckpt_step)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ddpm, loss_reg, loss_rgm, is_null_batch = trainer(
                    x_0, y_0, augm,
                    force_null_batch=(FLAGS.smoke_force_null_update == step + 1),
                    closure_calibrated_beta=calibrated_beta_now)
                trainer_core = trainer.module if FLAGS.parallel else trainer
                closure_stats = {
                    'closure_defect_raw': trainer_core.last_closure_defect.float(),
                    'closure_target_grad_norm': trainer_core.last_closure_target_grad_norm.float(),
                    'closure_actual_grad_norm': trainer_core.last_closure_actual_grad_norm.float(),
                    'closure_kkt_residual': trainer_core.last_closure_kkt_residual.float(),
                    'closure_simplex_error': trainer_core.last_closure_simplex_error.float(),
                    'closure_active_size': trainer_core.last_closure_active_size.float(),
                    'closure_teacher_calls': float(trainer_core.last_closure_teacher_calls),
                    'closure_projection_calls': float(trainer_core.last_closure_projection_calls),
                    'closure_calibrated_beta': trainer_core.last_closure_calibrated_beta.float(),
                    'closure_D_local': trainer_core.last_closure_local_defect.float(),
                    'closure_D_cal': trainer_core.last_closure_calibrated_defect.float(),
                    'closure_calibrated_gap': trainer_core.last_closure_calibrated_gap.float(),
                    'closure_normalized_gap': trainer_core.last_closure_normalized_gap.float(),
                    'closure_conditional_spread': trainer_core.last_closure_conditional_spread.float(),
                    'closure_calibrated_constraint_error': trainer_core.last_closure_calibrated_constraint_error.float(),
                    'closure_calibrated_fixed_residual': trainer_core.last_closure_calibrated_fixed_residual.float(),
                    'esc_used_low_t': float(trainer_core.last_used_low_t),
                    'esc_low_t_multiplier': float(trainer_core.last_nbt_multiplier),
                }
                if is_null_batch and FLAGS.bnt_alpha > 0:
                    # BNT/CANB: reweight null-branch batch by inverse class frequency
                    # and optional class-confusability pressure.
                    w_per_sample = bnt_w[y_0]  # [B] — use original y_0 before null override
                    loss_ddpm = (loss_ddpm.mean(dim=[1, 2, 3]) * w_per_sample).mean()
                elif FLAGS.cidr_beta > 0:
                    # CIDR: apply per-sample CIDR weight before averaging
                    w_per_sample = cidr_w[y_0]  # [B]
                    loss_ddpm = (loss_ddpm.mean(dim=[1, 2, 3]) * w_per_sample).mean()
                else:
                    loss_ddpm = loss_ddpm.mean()
                loss_reg = loss_reg.mean()
                loss = loss_ddpm + loss_reg if FLAGS.cb and loss_reg > 0 else loss_ddpm
                loss = loss + loss_rgm
                if is_null_batch:
                    closure_batches.append({
                        'update': step + 1,
                        'loss': float(loss.detach().float().cpu()),
                        'closure_mse': float(trainer_core.last_closure_defect.detach().float().cpu()),
                        'target_grad_norm': float(trainer_core.last_closure_target_grad_norm.detach().float().cpu()),
                        'actual_grad_norm': float(trainer_core.last_closure_actual_grad_norm.detach().float().cpu()),
                        'kkt_residual': float(trainer_core.last_closure_kkt_residual.detach().float().cpu()),
                        'simplex_error': float(trainer_core.last_closure_simplex_error.detach().float().cpu()),
                        'teacher_calls': int(trainer_core.last_closure_teacher_calls),
                        'projection_calls': int(trainer_core.last_closure_projection_calls),
                        'calibrated_beta': float(trainer_core.last_closure_calibrated_beta.detach().float().cpu()),
                        'D_local': float(trainer_core.last_closure_local_defect.detach().float().cpu()),
                        'D_cal': float(trainer_core.last_closure_calibrated_defect.detach().float().cpu()),
                        'calibrated_gap': float(trainer_core.last_closure_calibrated_gap.detach().float().cpu()),
                        'normalized_gap': float(trainer_core.last_closure_normalized_gap.detach().float().cpu()),
                        'conditional_spread': float(trainer_core.last_closure_conditional_spread.detach().float().cpu()),
                        'calibrated_constraint_error': float(trainer_core.last_closure_calibrated_constraint_error.detach().float().cpu()),
                        'calibrated_fixed_residual': float(trainer_core.last_closure_calibrated_fixed_residual.detach().float().cpu()),
                    })
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # logs
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('loss_ddpm', loss_ddpm, step)
            writer.add_scalar('loss_reg', loss_reg, step)
            writer.add_scalar('loss_rgm', loss_rgm, step)
            for name, value in closure_stats.items():
                writer.add_scalar(name, value, step)
            pbar.set_postfix(loss='%.5f' % loss)
            wandb_payload = {
                "loss": loss.item(),
                "loss_ddpm": loss_ddpm.item(),
                "loss_reg": loss_reg.item(),
                "loss_rgm": loss_rgm.item(),
            }
            wandb_payload.update({
                name: value.item() if torch.is_tensor(value) else value
                for name, value in closure_stats.items()
            })
            wandb.log(wandb_payload, step=step)

            # sample
            if step != FLAGS.ckpt_step and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0, _  = ema_sampler(fixed_x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # `step` is zero-indexed, while checkpoints name completed optimizer
            # updates.  This makes --total_steps=50000 mean exactly 50,000 updates
            # and emit ckpt_50000.pt rather than an off-by-one checkpoint.
            completed_updates = step + 1
            if FLAGS.save_step > 0 and completed_updates % FLAGS.save_step == 0:
                ckpt = build_native_checkpoint_payload(
                    net_model=net_model, ema_model=ema_model, optim=optim,
                    sched=sched, fixed_x_T=fixed_x_T,
                    dataloader_progress=datalooper, step=completed_updates)
                ckpt['method_config'] = method_config
                ckpt['resume_source'] = resume_source
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(completed_updates)))
                prev_ckpt = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(completed_updates - FLAGS.save_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                # net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID = evaluate(ema_sampler, ema_model, False)
                metrics = {
                    'IS': ema_IS[0],
                    'IS_std': ema_IS[1],
                    'FID': ema_FID
                }
                print(step, metrics)
                pbar.write(
                    '%d/%d ' % (step, FLAGS.total_steps) +
                    ', '.join('%s:%.5f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + '\n')
    writer.close()
    elapsed = time.monotonic() - started_at
    invocation_updates = FLAGS.total_steps - FLAGS.ckpt_step
    checkpoint = os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(FLAGS.total_steps))
    if not os.path.isfile(checkpoint):
        raise RuntimeError('native final checkpoint is missing; runtime evidence withheld')
    summary = {
        'schema': 'crt-c1-runtime-v1', 'status': 'completed', 'exit_code': 0,
        'protocol_mode': FLAGS.protocol_mode, 'arm': FLAGS.runtime_arm,
        'requested_updates': invocation_updates, 'completed_updates': invocation_updates,
        'native_checkpoint_step': FLAGS.total_steps, 'checkpoint': checkpoint,
        'observed_null_batches': len(closure_batches), 'closure_batches': closure_batches,
        'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
        'elapsed_seconds': elapsed, 'updates_per_second': invocation_updates / elapsed,
        'seed': FLAGS.seed, 'batch_size': FLAGS.batch_size,
        'source_manifest_sha256': FLAGS.source_manifest_sha256,
        'protocol_sha256': FLAGS.protocol_sha256,
        'initial_model_state_sha256': initial_model_state_sha256,
        'method_config': method_config,
        'resume_source': resume_source,
    }
    runtime_path = os.path.join(FLAGS.logdir, 'RUNTIME_SUMMARY.json')
    try:
        fd = os.open(runtime_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise RuntimeError('runtime summary already exists') from exc
    with os.fdopen(fd, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')


def eval():
    set_global_seed()
    FLAGS.num_class = 100 if 'cifar100' in FLAGS.data_type else 10
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.num_class, FLAGS.img_size, FLAGS.var_type).to(device)

    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    FLAGS.sample_name = '{}_N{}_STEP{}'.format(FLAGS.sample_name, FLAGS.num_images, FLAGS.ckpt_step)

    # load ema model (almost always better than the model) and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt_{}.pt'.format(FLAGS.ckpt_step)), map_location='cpu')

    # evaluate IS/FID
    if 'cifar100' in FLAGS.data_type:
        FLAGS.fid_cache = './stats/cifar100.train.npz'
    else:
        FLAGS.fid_cache = './stats/cifar10.train.npz'

    if not FLAGS.sampled:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model = None

    (IS, IS_std), FID, prd_score, ipr = evaluate(sampler, model, FLAGS.sampled)

    print('logdir', FLAGS.logdir)
    print("Model(EMA): IS:%6.5f(%.5f), FID/CIFAR100:%7.5f \n" % (IS, IS_std, FID))
    print("Improved PRD:%6.5f, RECALL:%7.5f \n" % (ipr[0], ipr[1]))
    print("PRD PRECISION FOR 100 CLASSES:%6.5f, RECALL:%7.5f \n" % (prd_score[0], prd_score[1]))

    with open(os.path.join(FLAGS.logdir,  'res_ema_{}.txt'.format(FLAGS.sample_name)), 'a+') as f:
        f.write("Settings: NUM:{} EPOCH:{}, OMEGA:{}, METHOD:{} \n" .format (FLAGS.num_images, FLAGS.ckpt_step, FLAGS.omega,FLAGS.sample_method))
        f.write("Model(EMA): IS:%6.5f(%.5f), FID/CIFAR100:%7.5f \n" % (IS, IS_std, FID))
        f.write("Improved PRD:%6.5f, RECALL:%7.5f \n" % (ipr[0], ipr[1]))
        f.write("PRD PRECISION FOR 100 CLASSES:%6.5f, RECALL:%7.5f \n" % (prd_score[0], prd_score[1]))
    f.close()


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
