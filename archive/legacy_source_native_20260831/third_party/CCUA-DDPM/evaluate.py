import copy
import json
import os
import warnings
from absl import app, flags
from tqdm import trange

import torch
import numpy as np
from torchvision.datasets import ImageFolder
# import kagglehub


from torchvision.utils import make_grid, save_image
from torchvision import transforms

try:
    from tensorboardX import SummaryWriter
except Exception as err:
    pass
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, GaussianDiffusionSamplerDDIM
from model.model import UNet
from model.classifier import HalveUNetClassifier

from torchvision.datasets import Places365
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, ImbalanceImageNet, ImageNet, PlacesLT, PlacesLD, SubsetPerLabel

from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.KID import KID
from fld.metrics.PrecisionRecall import PrecisionRecall

from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor


FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 42, help='random seed')
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('resume', False, help='resume from a checkpoint')
flags.DEFINE_string('resume_dir', './', help='the resumed checkpoint')
flags.DEFINE_integer('ckpt_step', 0, help='resumed checkpoint step')
flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
flags.DEFINE_bool('sample', False, help='load model.pt and run inference')


# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')

flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_bool('conditional', False, help='conditional generation')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient norm clipping')
flags.DEFINE_integer('total_steps', 300001, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 64, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help='ema decay rate')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with with 10%  probability')

# Dataset
flags.DEFINE_string('data_type', 'cifar10lt',
                    help='data type, must be in [cifar10lt, cifar100lt, imgnetlt, tinyimgnetlt, placeslt, placeslt-10]')
flags.DEFINE_string('data_path', '.cache/data', help='data path')
flags.DEFINE_float('imb_factor', 0.01, help='imb_factor for long tail dataset')
flags.DEFINE_integer('num_class', 0, help='number of class of the pretrained model')

# Logging & Sampling
flags.DEFINE_string('logdir', './logs/', help='log directory')
flags.DEFINE_integer('sample_size', 64, 'sampling size of images')
flags.DEFINE_integer('sample_step', 50000, help='frequency of sampling')
flags.DEFINE_string('sample_method', 'cfg', help='sampling method, must be in [ddim, ddpm / cfg, uncond]')
flags.DEFINE_float('omega', 1.5, help='guidance strength')
flags.DEFINE_string('omega_scheduler', 'constant', help='omega scheduler')
flags.DEFINE_float('gamma', 100, help='only works when omega_scheduler is "gamma"')
# DDIM Sampling Related
flags.DEFINE_integer('ddim_skip_step', 10, help="ddim step")

# Evaluation
flags.DEFINE_integer('save_step', 100000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_string('sample_name', 'saved', help='name for a set of samples to be saved or to be evaluated')
flags.DEFINE_bool('sampled', False, help='evaluate sampled images')
flags.DEFINE_string('metrics', 'all', help='fid | fid_clip ifid | fid_tail | fld | all')
# Used Parameters
flags.DEFINE_string('category', 'all', help='already deprecated, just used for alignment with main.py')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './', help='FID cache')
flags.DEFINE_string('feats_cache', './', help='FEATS cache')
flags.DEFINE_bool('prd', False, help='evaluate precision and recall (F_beta), only evaluated with 50k samples')
flags.DEFINE_bool('improved_prd', False, help='evaluate improved precision and recall, only evaluated with 50k samples')


# SOTA Methods
# OCLT hyperparameter
flags.DEFINE_bool('transfer_x0', False, help='transfering x0 to other index based on L2 norm')
flags.DEFINE_bool('transfer_tr_tau', False, help='transfering x0 with adjusted tau')
flags.DEFINE_bool('transfer_mixing', False, help='whether to using transfer')
flags.DEFINE_bool('bal_sample', False, help='whether to using transfer')
flags.DEFINE_string('transfer_mode', 'full', help='transfer_mode')
flags.DEFINE_float('tr_tau', 1.0, help='weight for transfer power')

# CBDM hyperparameter
flags.DEFINE_bool('cbdm', False, help='training with CBDM')
flags.DEFINE_float('cb_tau', 1.0, help='temperature for CBDM')

# CCUA hyperparameter
flags.DEFINE_float('ccua_al', 0.0, help='ccua al loss')
flags.DEFINE_float('ccua_ucl', 0.0, help='ccua ucl loss')
flags.DEFINE_bool('brs', False, help='whether to use batch resample')
flags.DEFINE_float('brs_factor', 0.1, help='imb_factor after batch resample')


device = torch.device('cuda:0')


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, num_class=FLAGS.num_class)

    sampler = GaussianDiffusionSamplerDDIM(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        var_type=FLAGS.var_type, omega=FLAGS.omega,
        omega_scheduler=FLAGS.omega_scheduler, gamma=FLAGS.gamma, cond=FLAGS.conditional).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    FLAGS.sample_name = '{}_N{}_STEP{}_Omega{}_OmegaScheduler{}_Gamma{}_SEED{}_DDIM{}steps'.format(
        FLAGS.sample_name,
        FLAGS.num_images,
        FLAGS.ckpt_step,
        FLAGS.omega,
        FLAGS.omega_scheduler, FLAGS.gamma,
        FLAGS.seed, 1000 // FLAGS.ddim_skip_step)

    # load model and evaluate
    if FLAGS.ckpt_step >= 0:
        ckpt = torch.load(os.path.join(FLAGS.logdir, f'ckpt_{FLAGS.ckpt_step}.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))

    model.load_state_dict(ckpt['net_model'])
    model.load_state_dict(ckpt['ema_model'])

    if not FLAGS.sampled:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model = None

    metrcis_log = evaluate_metrics(sampler, model, FLAGS.sampled)

    with open(os.path.join(FLAGS.logdir, 'res_ema_{}.txt'.format(FLAGS.sample_name)), 'a+') as f:
        f.write(metrcis_log)
        f.close()


def evaluate_metrics(sampler, model, sampled=False):
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([FLAGS.img_size, FLAGS.img_size], antialias=True),
        transforms.ToPILImage()
    ])

    if FLAGS.data_type == 'cifar10lt':
        bal_trainset = ImbalanceCIFAR10(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=1.0,
            rand_number=0,
            train=True,
            transform=eval_transform,
            target_transform=None,
            download=True,
        )
        bal_valset = ImbalanceCIFAR10(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=1.0,
            rand_number=0,
            train=False,
            transform=eval_transform,
            target_transform=None,
            download=True,
        )
        lt_trainset = ImbalanceCIFAR10(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=FLAGS.imb_factor,
            rand_number=0,
            train=True,
            transform=eval_transform,
            target_transform=None,
            download=True,
        )
    elif FLAGS.data_type == 'cifar100lt':
        bal_trainset = ImbalanceCIFAR100(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=1.0,
            rand_number=0,
            train=True,
            transform=eval_transform,
            download=True,
        )
        bal_valset = ImbalanceCIFAR100(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=1.0,
            rand_number=0,
            train=False,
            transform=eval_transform,
            download=True,
        )
        lt_trainset = ImbalanceCIFAR100(
            root=FLAGS.data_path,
            imb_type='exp',
            imb_factor=FLAGS.imb_factor,
            rand_number=0,
            train=True,
            transform=eval_transform,
            download=True,
        )
    elif FLAGS.data_type == 'imgnetlt' or FLAGS.data_type == 'tinyimgnetlt':
        bal_trainset = ImageNet(root=FLAGS.data_path,
                                split="train",
                                transform=eval_transform
                                )
        bal_valset = ImageNet(root=FLAGS.data_path,
                              split="val",
                              transform=eval_transform
                              )
        lt_trainset = ImbalanceImageNet(root=FLAGS.data_path,
                                    split="train",
                                    imb_type='exp',
                                    imb_factor=FLAGS.imb_factor,
                                    rand_number=0,
                                    transform=eval_transform
                                    )
    elif FLAGS.data_type == 'placeslt':
        # bal_trainset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
        #                         split='train-standard', small=True,
        #                         imb_type="exp",
        #                         imb_factor=1.0,
        #                         rand_number=0,
        #                         transform=eval_transform,
        #                         download=True)
        # bal_valset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
        #                       split='val', small=True,
        #                       imb_type="exp",
        #                       imb_factor=1.0,
        #                       rand_number=0,
        #                       transform=eval_transform,
        #                       download=True)
        bal_trainset = Places365(root=FLAGS.data_path,  # /data/datasets/Places/',
                                 split='train-standard', small=True,
                                 transform=eval_transform,
                                 download=False,
                                 )
        bal_valset = Places365(root=FLAGS.data_path,  # /data/datasets/Places/',
                               split='val', small=True,
                               transform=eval_transform,
                               download=False,
                               )
        # bal_trainset = bal_valset
        lt_trainset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
                           split='train-standard', small=True,
                           imb_type="exp",
                           imb_factor=FLAGS.imb_factor,
                           rand_number=0,
                           transform=eval_transform,
                           download=False)
    elif FLAGS.data_type == 'placeslt-10':
        # bal_trainset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
        #                         split='train-standard', small=True,
        #                         imb_type="exp",
        #                         imb_factor=1.0,
        #                         rand_number=0,
        #                         transform=eval_transform,
        #                         download=True)
        # bal_valset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
        #                       split='val', small=True,
        #                       imb_type="exp",
        #                       imb_factor=1.0,
        #                       rand_number=0,
        #                       transform=eval_transform,
        #                       download=True)
        bal_trainset = Places365(root=FLAGS.data_path,  # /data/datasets/Places/',
                                 split='train-standard', small=True,
                                 transform=eval_transform,
                                 download=True,
                                 )
        bal_valset = Places365(root=FLAGS.data_path,  # /data/datasets/Places/',
                               split='val', small=True,
                               transform=eval_transform,
                               download=True,
                               )
        bal_trainset = SubsetPerLabel(bal_trainset, label_indices=[0, 1, 2, 99, 100, 199, 200, 362, 363, 364])
        bal_valset = SubsetPerLabel(bal_valset, label_indices=[0,1,2, 99,100,199,200, 362,363,364])
        # bal_trainset = bal_valset
        lt_trainset = PlacesLT(root=FLAGS.data_path,  # /data/datasets/Places/',
                           split='train-standard', small=True,
                           imb_type="exp",
                           imb_factor=FLAGS.imb_factor,
                           rand_number=0,
                           transform=eval_transform,
                           download=True)
        lt_trainset = SubsetPerLabel(lt_trainset, label_indices=[0,1,2, 99,100,199,200, 362,363,364])
    elif FLAGS.data_type == 'placesld-10':
        bal_trainset = PlacesLD(root=FLAGS.data_path,  # /data/datasets/Places/',
                                split='train-standard', small=True,
                                img_num_per_cls=10,
                                rand_number=0,
                                transform=eval_transform,
                                download=True)
        bal_trainset = SubsetPerLabel(bal_trainset, label_indices=[0, 1, 2, 99, 100, 199, 200, 362, 363, 364])
        bal_valset = PlacesLD(root=FLAGS.data_path,  # /data/datasets/Places/',
                              split='val', small=True,
                              img_num_per_cls=10,
                              rand_number=0,
                              transform=eval_transform,
                              download=True)
        bal_valset = SubsetPerLabel(bal_valset, label_indices=[0,1,2, 99,100,199,200, 362,363,364])
        # bal_trainset = bal_valset
        lt_trainset = PlacesLD(root=FLAGS.data_path,  # /data/datasets/Places/',
                               split='train-standard', small=True,
                               img_num_per_cls=10,
                               rand_number=0,
                               transform=eval_transform,
                               download=True)
        lt_trainset = SubsetPerLabel(lt_trainset, label_indices=[0,1,2, 99,100,199,200, 362,363,364])
    else:
        raise NotImplementedError('Please enter a valid data type!!!.')

    print("len_train_bal: {}, len_val_bal: {}, len_train_lt: {}".format(len(bal_trainset), len(bal_valset), len(lt_trainset)))
    print('sampled: ', sampled)
    if not sampled:
        model.eval()
        with torch.no_grad():
            images = []
            labels = []
            desc = 'generating images'
            for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
                batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
                x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
                # batch_idx = torch.randint(len(classes), size=(x_T.shape[0],))
                # batch_labels = classes[batch_idx].to(device)
                batch_labels = torch.randint(FLAGS.num_class, size=(x_T.shape[0],)).to(device)
                print(f'label len: {batch_labels.shape[0]}, label range: {batch_labels.min()} ~ {batch_labels.max()}')
                batch_images = sampler(x_T.to(device), y=batch_labels.to(device),
                                       method=FLAGS.sample_method,
                                       skip=FLAGS.ddim_skip_step,
                                       return_timestep_status=False)
                images.append((batch_images + 1) / 2)
                if FLAGS.sample_method != 'uncond' and batch_labels is not None:
                    labels.append(batch_labels)
            images = torch.cat(images, dim=0).cpu().numpy()
        np.save(os.path.join(FLAGS.logdir, '{}_{}_samples_ema_{}.npy'.format(
            FLAGS.sample_method, FLAGS.omega,
            FLAGS.sample_name)), images)
        if FLAGS.sample_method != 'uncond':
            labels = torch.cat(labels, dim=0).cpu().numpy()
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

    # images = sorted(images, key=lambda x: x.mean(), reverse=True)
    save_image(
        torch.tensor(images[:256]),
        os.path.join(FLAGS.logdir, 'visual_ema_{}_{}_{}.png'.format(
            FLAGS.sample_method, FLAGS.omega, FLAGS.sample_name)),
        nrow=16)

    incept_feature_extractor = InceptionFeatureExtractor()
    clip_feature_extractor = CLIPFeatureExtractor()
    # dinov2_feature_extractor = DINOv2FeatureExtractor()

    gen_images = torch.tensor(images)

    metrics = FLAGS.metrics.lower().replace(' ', '').split(',')


    log_out = ""

    if 'fid' in metrics or 'kid' in metrics \
        or 'all' in metrics:
        incept_train_feat = incept_feature_extractor.get_features(bal_trainset)
        incept_val_feat = incept_feature_extractor.get_features(bal_valset)
        incept_gen_feat = incept_feature_extractor.get_tensor_features(gen_images)

        fid = FID().compute_metric(incept_train_feat, None, incept_gen_feat)
        print('Trainset FID: {:.5f}'.format(fid))
        log_out += 'Trainset FID: {:.5f}\n'.format(fid)

        fid = FID().compute_metric(incept_val_feat, None, incept_gen_feat)
        print('Valset FID: {:.5f}'.format(fid))
        log_out += 'Valset FID: {:.5f}\n'.format(fid)

        kid = KID().compute_metric(incept_train_feat, None, incept_gen_feat)
        print('Trainset KID: {:.5f}'.format(kid))
        log_out += 'Trainset KID: {:.5f}\n'.format(kid)

        kid = KID().compute_metric(incept_val_feat, None, incept_gen_feat)
        print('Valset KID: {:.5f}'.format(kid))
        log_out += 'Valset KID: {:.5f}\n'.format(kid)

    if 'precision' in metrics or 'recall' in metrics \
        or 'all' in metrics:
        incept_train_feat = incept_feature_extractor.get_features(bal_trainset)
        incept_val_feat = incept_feature_extractor.get_features(bal_valset)
        incept_gen_feat = incept_feature_extractor.get_tensor_features(gen_images)

        precision = PrecisionRecall(mode="Precision").compute_metric(incept_train_feat, None, incept_gen_feat)
        print('Trainset Precision: {:.5f}'.format(precision))
        log_out += 'Trainset Precision: {:.5f}\n'.format(precision)

        precision = PrecisionRecall(mode="Precision").compute_metric(incept_val_feat, None, incept_gen_feat)
        print('Valset Precision: {:.5f}'.format(precision))
        log_out += 'Valset Precision: {:.5f}\n'.format(precision)

        recall = PrecisionRecall(mode="Recall").compute_metric(incept_train_feat, None, incept_gen_feat)
        print('Trainset Recall: {:.5f}'.format(recall))
        log_out += 'Trainset Recall: {:.5f}\n'.format(recall)

        recall = PrecisionRecall(mode="Recall").compute_metric(incept_val_feat, None, incept_gen_feat)
        print('Valset Recall: {:.5f}'.format(recall))
        log_out += 'Valset Recall: {:.5f}\n'.format(recall)

    if 'fid_clip' in metrics or 'all' in metrics:
        clip_train_feat = clip_feature_extractor.get_features(bal_trainset)
        clip_val_feat = clip_feature_extractor.get_features(bal_valset)
        clip_gen_feat = clip_feature_extractor.get_tensor_features(gen_images)

        fid_clip = FID().compute_metric(clip_train_feat, None, clip_gen_feat)
        print('Trainset FID_CLIP: {:.5f}'.format(fid_clip))
        log_out += 'Trainset FID_CLIP: {:.5f}\n'.format(fid_clip)

        fid_clip = FID().compute_metric(clip_val_feat, None, clip_gen_feat)
        print('Valset FID_CLIP: {:.5f}'.format(fid_clip))
        log_out += 'Valset FID_CLIP: {:.5f}\n'.format(fid_clip)

    # if 'fld' in metrics or 'all' in metrics:
    #     dinov2_train_feat = dinov2_feature_extractor.get_features(lt_trainset)
    #     dinov2_val_feat = dinov2_feature_extractor.get_features(bal_valset)
    #     dinov2_gen_feat = dinov2_feature_extractor.get_tensor_features(gen_images)
    #
    #     fld = FLD().compute_metric(dinov2_train_feat, dinov2_val_feat, dinov2_gen_feat)
    #     fld_gap = FLD(eval_feat="gap").compute_metric(dinov2_train_feat, dinov2_val_feat, dinov2_gen_feat)
    #     print('FLD: {:.5f}, Generalization Gap FLD: {:.5f}'.format(fld, fld_gap))
    #     log_out += 'FLD: {:.5f}, Generalization Gap FLD: {:.5f}\n'.format(fld, fld_gap)

    if 'ifid' in metrics or 'all' in metrics:
        def ifids_metrics(mode='trainset'):
            ifids = []
            for i in range(FLAGS.num_class):
                if mode == 'trainset':
                    sub_refset = SubsetPerLabel(bal_trainset, label_indices=[i])
                elif mode == 'valset':
                    sub_refset = SubsetPerLabel(bal_valset, label_indices=[i])
                else:
                    raise ValueError('Unknown dataset mode: {}'.format(mode))

                sub_genimages = torch.tensor(images[labels==i])

                i_ref_feat = incept_feature_extractor.get_features(sub_refset)
                i_gen_feat = incept_feature_extractor.get_tensor_features(sub_genimages)

                ifid = FID().compute_metric(i_ref_feat, None, i_gen_feat)
                ifids.append(round(ifid, 5))
            ifids = np.array(ifids)
            return ifids

        ifids = ifids_metrics('trainset')
        print('Trainset iFIDs: {}, iFID: {:.5f}'.format(ifids, ifids.mean()))
        log_out += 'Trainset iFIDs: {}, iFID: {:.5f}\n'.format(ifids, ifids.mean())

        ifids = ifids_metrics('valset')
        print('Valset iFIDs: {}, iFID: {:.5f}'.format(ifids, ifids.mean()))
        log_out += 'Valset iFIDs: {}, iFID: {:.5f}\n'.format(ifids, ifids.mean())

    def subset_feat(classes, include_valset=False):
        sub_trainset = SubsetPerLabel(bal_trainset, label_indices=classes)
        sub_train_feat = incept_feature_extractor.get_features(sub_trainset)

        sub_val_feat = None
        if include_valset:
            sub_valset = SubsetPerLabel(bal_valset, label_indices=classes)
            sub_val_feat = incept_feature_extractor.get_features(sub_valset)

        indices = []
        for i in classes:
            indices.append(np.where(labels == i))
        indices = np.hstack(indices).reshape(-1)
        sub_genimages = torch.tensor(images[indices])

        sub_gen_feat = incept_feature_extractor.get_tensor_features(sub_genimages)
        return sub_train_feat, sub_val_feat, sub_gen_feat

    if 'fid_head' in metrics or 'all' in metrics:
        classes = torch.arange(0, FLAGS.num_class // 3).numpy().tolist()
        sub_train_feat, sub_val_feat, sub_gen_feat = subset_feat(classes, include_valset=True)

        fid_sub = FID().compute_metric(sub_train_feat, None, sub_gen_feat)
        print('Trainset FID_head: {:.5f}'.format(fid_sub))
        log_out += 'Trainset FID_head: {:.5f}\n'.format(fid_sub)

        fid_sub = FID().compute_metric(sub_val_feat, None, sub_gen_feat)
        print('Valset FID_head: {:.5f}'.format(fid_sub))
        log_out += 'Valset FID_head: {:.5f}\n'.format(fid_sub)

    if 'fid_body' in metrics or 'all' in metrics:
        classes = torch.arange(1 + FLAGS.num_class // 3, 2 * FLAGS.num_class // 3).numpy().tolist()
        sub_train_feat, sub_val_feat, sub_gen_feat = subset_feat(classes, include_valset=True)

        fid_sub = FID().compute_metric(sub_train_feat, None, sub_gen_feat)
        print('Trainset FID_body: {:.5f}'.format(fid_sub))
        log_out += 'Trainset FID_body: {:.5f}\n'.format(fid_sub)

        fid_sub = FID().compute_metric(sub_val_feat, None, sub_gen_feat)
        print('Valset FID_body: {:.5f}'.format(fid_sub))
        log_out += 'Valset FID_body: {:.5f}\n'.format(fid_sub)

    if 'fid_tail' in metrics or 'all' in metrics:
        classes = torch.arange(1 + 2 * FLAGS.num_class // 3, FLAGS.num_class).numpy().tolist()
        sub_train_feat, sub_val_feat, sub_gen_feat = subset_feat(classes, include_valset=True)

        fid_sub = FID().compute_metric(sub_train_feat, None, sub_gen_feat)
        print('Trainset FID_tail: {:.5f}'.format(fid_sub))
        log_out += 'Trainset FID_tail: {:.5f}\n'.format(fid_sub)

        fid_sub = FID().compute_metric(sub_val_feat, None, sub_gen_feat)
        print('Valset FID_tail: {:.5f}'.format(fid_sub))
        log_out += 'Valset FID_tail: {:.5f}\n'.format(fid_sub)

    if 'fid_head_10' in metrics or 'all' in metrics:
        classes = torch.arange(0, 10).numpy().tolist()
        sub_train_feat, sub_val_feat, sub_gen_feat = subset_feat(classes, include_valset=True)

        fid_sub = FID().compute_metric(sub_train_feat, None, sub_gen_feat)
        print('Trainset FID_head_10: {:.5f}'.format(fid_sub))
        log_out += 'Trainset FID_head_10: {:.5f}\n'.format(fid_sub)

        fid_sub = FID().compute_metric(sub_val_feat, None, sub_gen_feat)
        print('Valset FID_head_10: {:.5f}'.format(fid_sub))
        log_out += 'Valset FID_head_10: {:.5f}\n'.format(fid_sub)

    if 'fid_tail_10' in metrics or 'all' in metrics:
        classes = torch.arange(FLAGS.num_class - 10, FLAGS.num_class).numpy().tolist()
        sub_train_feat, sub_val_feat, sub_gen_feat = subset_feat(classes, include_valset=True)

        fid_sub = FID().compute_metric(sub_train_feat, None, sub_gen_feat)
        print('Trainset FID_tail_10: {:.5f}'.format(fid_sub))
        log_out += 'Trainset FID_tail_10: {:.5f}\n'.format(fid_sub)

        fid_sub = FID().compute_metric(sub_val_feat, None, sub_gen_feat)
        print('Valset FID_tail_10: {:.5f}'.format(fid_sub))
        log_out += 'Valset FID_tail_10: {:.5f}\n'.format(fid_sub)

    # if 'recall_tail' in metrics or 'all' in metrics:
    #     tail_classes = torch.arange(1 + 2 * FLAGS.num_class // 3, FLAGS.num_class).numpy().tolist()
    #     sub_trainset = SubsetPerLabel(bal_trainset, label_indices=tail_classes)
    #     indices = []
    #     for i in tail_classes:
    #         indices.append(np.where(labels == i))
    #     indices = np.hstack(indices).reshape(-1)
    #     sub_genimages = torch.tensor(images[indices])
    #
    #     sub_train_feat = incept_feature_extractor.get_features(sub_trainset)
    #     sub_gen_feat = incept_feature_extractor.get_tensor_features(sub_genimages)
    #
    #     recall_tail = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(sub_train_feat, None, sub_gen_feat)
    #     print('Recall_tail: {:.5f}'.format(recall_tail))
    #     log_out += 'Recall_tail: {:.5f}\n'.format(recall_tail)
    #
    # if 'recall_tail_10' in metrics or 'all' in metrics:
    #     tail_classes = torch.arange(FLAGS.num_class - 10, FLAGS.num_class).numpy().tolist()
    #     sub_trainset = SubsetPerLabel(bal_trainset, label_indices=tail_classes)
    #     indices = []
    #     for i in tail_classes:
    #         indices.append(np.where(labels == i))
    #     indices = np.hstack(indices).reshape(-1)
    #     sub_genimages = torch.tensor(images[indices])
    #
    #     sub_train_feat = incept_feature_extractor.get_features(sub_trainset)
    #     sub_gen_feat = incept_feature_extractor.get_tensor_features(sub_genimages)
    #
    #     recall_tail = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(sub_train_feat, None, sub_gen_feat)
    #     print('Recall_tail_10: {:.5f}'.format(recall_tail))
    #     log_out += 'Recall_tail_10: {:.5f}\n'.format(recall_tail)

    # if 'ifld' in metrics or 'all' in metrics:
    #     iflds = []
    #     for i in range(FLAGS.num_class):
    #         sub_trainset = SubsetPerLabel(lt_trainset, label_indices=[i])
    #         sub_valset = SubsetPerLabel(bal_valset, label_indices=[i])
    #         sub_genimages = torch.tensor(images[labels==i])
    #
    #         i_train_feat = dinov2_feature_extractor.get_features(sub_trainset)
    #         i_val_feat = dinov2_feature_extractor.get_features(sub_valset)
    #         i_gen_feat = dinov2_feature_extractor.get_tensor_features(sub_genimages)
    #
    #         ifld = FLD(eval_feat="test", gen_size=len(sub_trainset)).compute_metric(i_train_feat, i_val_feat, i_gen_feat)
    #         iflds.append(round(ifld, 5))
    #     iflds = np.array(iflds)
    #
    #     print('iFLDs: {}, iFLD: {:.5f}'.format(iflds, iflds.mean()))
    #     log_out += 'iFLDs: {}, iFLD: {:.5f}\n'.format(iflds, iflds.mean())

    return log_out


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        torch.cuda.manual_seed_all(FLAGS.seed)

    if FLAGS.eval:
        FLAGS.brs = False
        assert FLAGS.brs == False

        print('Evaluating...')
        print(f'Image Size: {FLAGS.img_size}')
        eval()


if __name__ == '__main__':
    app.run(main)
