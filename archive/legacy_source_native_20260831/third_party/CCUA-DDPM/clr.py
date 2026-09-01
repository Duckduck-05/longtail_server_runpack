import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from info_nce import InfoNCE, info_nce

def cond_info_nce(f, y, reduction='none'):
    queries = rearrange(f, 'b c h w -> b (c h w)')
    # print('queries: ', queries.shape, '\ty_0: ', y_0.shape)
    bs, d = queries.shape
    num_negative_keys = bs
    positive_keys = torch.zeros(bs, d).to(queries.device)
    negative_keys = torch.zeros(bs, num_negative_keys, d).to(queries.device)
    for i in range(bs):
        label = y.detach().cpu().numpy()
        pos_idx = np.where(label == label[i])[0]
        if pos_idx.size > 0:
            pos_idx = np.random.choice(pos_idx, 1)
        else:
            pos_idx = i
        positive_keys[i] = queries[pos_idx]

        neg_idx = np.where(label != label[i])[0]
        if neg_idx.size > 0:
            neg_idx = np.random.choice(neg_idx, num_negative_keys, replace=True)
            negative_keys[i] = queries[neg_idx]
    loss_nce = info_nce(queries, positive_keys, negative_keys, reduction=reduction, negative_mode='paired')
    # print('positive_keys: ', positive_keys.shape, '\tnegative_keys: ', negative_keys.shape, '\tnce_loss: ', loss_nce)
    return loss_nce

def uncond_info_nce(fq, fk, reduction='none'):
    queries = rearrange(fq, 'b c h w -> b (c h w)')
    keys = rearrange(fk, 'b c h w -> b (c h w)')
    bs, d = keys.shape
    num_negative_keys = bs - 1
    positive_keys = torch.zeros(bs, d).to(keys.device)
    negative_keys = torch.zeros(bs, num_negative_keys, d).to(keys.device)
    for i in range(bs):
        positive_keys[i] = keys[i]
        negative_keys[i] = torch.cat([keys[:i], keys[i+1:]], dim=0)
    loss_nce = info_nce(queries, positive_keys, negative_keys, reduction=reduction, negative_mode='paired')
    return loss_nce


# class CLRQueueLoss(nn.Module):
#
#     def __init__(self, dim, K=65536, T=0.07): # K=1024, T=0.1 for CIFAR10LT
#         """
#         dim: feature dimension
#         K: queue size; number of negative keys (default: 65536)
#         T: softmax temperature (default: 0.07)
#         """
#         super(CLRQueueLoss, self).__init__()
#
#         self.K = K
#         self.T = T
#
#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         # gather keys before updating queue
#         # keys = concat_all_gather(keys)
#
#         batch_size = keys.shape[0]
#
#         ptr = int(self.queue_ptr)
#         assert self.K % batch_size == 0  # for simplicity
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr : ptr + batch_size] = keys.T
#         ptr = (ptr + batch_size) % self.K  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     @torch.no_grad()
#     def _batch_shuffle(self, x):
#         batch_size = x.shape[0]
#         idx_shuffle = torch.randperm(batch_size).cuda()
#         idx_unshuffle = torch.argsort(idx_shuffle)
#         return x[idx_shuffle], idx_unshuffle
#
#     @torch.no_grad()
#     def _batch_unshuffle(self, x, idx_unshuffle):
#         return x[idx_unshuffle]
#
#     def forward(self, fq, fk, reduction='none'):
#         """
#         Input:
#             fq: a batch of features
#             fk: a batch of features
#         Output:
#             logits, targets
#         """
#         q = rearrange(fq, 'b c h w -> b (c h w)')
#         k = rearrange(fk, 'b c h w -> b (c h w)')
#
#         # # compute query features
#         q = nn.functional.normalize(q, dim=1)
#         #
#         # k, idx_unshuffle = self._batch_shuffle(k)
#         k = nn.functional.normalize(k, dim=1)
#         # k = self._batch_unshuffle(k, idx_unshuffle)
#
#         # compute logits
#         # Einstein sum is more intuitive
#         # positive logits: Nx1
#         l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
#         # negative logits: NxK
#         l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
#
#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1)
#
#         # apply temperature
#         logits /= self.T
#
#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#
#         # dequeue and enqueue
#         self._dequeue_and_enqueue(k)
#
#         return nn.CrossEntropyLoss(reduction=reduction)(logits, labels) * (2 * self.T)
