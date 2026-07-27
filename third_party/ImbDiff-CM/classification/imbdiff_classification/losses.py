import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


class LALoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, LA_epoch=-1):
        super().__init__()
        self.tau = tau
        self.LA_epoch = LA_epoch
        self.need_la = True
        prior = torch.tensor(cls_num_list, dtype=torch.float)
        self.register_buffer("class_prior", prior / prior.sum())

    def set_epoch(self, epoch):
        self.need_la = True if self.LA_epoch == -1 else epoch > self.LA_epoch

    def forward(self, outputs, target):
        logits = outputs["output"] if isinstance(outputs, dict) else outputs
        if self.need_la:
            logits = logits + self.tau * torch.log(self.class_prior + EPS)
        return F.cross_entropy(logits, target)


class CMLoss(nn.Module):
    def __init__(
        self,
        cls_num_list,
        tau=1.0,
        LA_epoch=-1,
        alpha=1.0,
        T=1.5,
        w_con=1.0,
        w_div=1.0,
    ):
        super().__init__()
        self.tau = tau
        self.LA_epoch = LA_epoch
        self.need_la = True
        self.alpha = alpha
        self.T = T
        self.w_con = w_con
        self.w_div = w_div
        prior = torch.tensor(cls_num_list, dtype=torch.float)
        prior = prior / prior.sum()
        inverse = 1 / prior
        inverse = inverse / inverse.sum()
        self.register_buffer("class_prior", prior)
        self.register_buffer("inverse_prior", inverse)

    def set_epoch(self, epoch):
        self.need_la = True if self.LA_epoch == -1 else epoch > self.LA_epoch

    def forward(self, outputs, target):
        logit_base, logit_cm = outputs["logits"]
        if self.need_la:
            logit_cm = logit_cm + self.tau * torch.log(self.class_prior + EPS)
        loss = F.cross_entropy(logit_cm, target)
        kl_div = F.kl_div(
            F.log_softmax(logit_base / self.T, dim=1),
            F.softmax(logit_cm / self.T, dim=1),
            reduction="none",
        ).mean(1)
        weight = self.class_prior[target] * self.w_con - self.inverse_prior[target] * self.w_div
        loss = loss + self.alpha * (kl_div * weight).mean() * logit_base.shape[1]
        return loss


def create_loss(name, params, cls_num_list):
    if name == "LALoss":
        return LALoss(cls_num_list=cls_num_list, **params)
    if name == "CMLoss":
        return CMLoss(cls_num_list=cls_num_list, **params)
    raise ValueError(f"Unsupported classification loss: {name}")

