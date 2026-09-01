import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

class Conv2d_LoRA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super(Conv2d_LoRA, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        if lora_mode == 'value':
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        elif lora_mode == 'ratio':
            self.r = max(int(r_ratio * min(in_channels, out_channels)), 1) if r_ratio > 0 else 0
            self.scaling = scaling
        else:
            raise ValueError('mode should be value or ratio')
        if self.r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros(self.r*kernel_size, in_channels*kernel_size))
            self.lora_B = nn.Parameter(self.weight.new_zeros(out_channels//self.groups*kernel_size, self.r*kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        super(Conv2d_LoRA, self).reset_parameters()
        if hasattr(self, 'r'):
            if self.r > 0:
                init.kaiming_normal_(self.lora_A, a = math.sqrt(5))
            # init.kaiming_normal_(self.lora_B, a = math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def forward(self, input, LoRA=True):
        if self.r > 0 and LoRA:
            weight = self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape)
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DownSample_CM(nn.Module):
    def __init__(self, in_ch, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super().__init__()
        self.main = Conv2d_LoRA(in_ch, in_ch, 3, stride=2, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, use_cm=True):
        x = self.main(x, use_cm)
        return x

class UpSample_CM(nn.Module):
    def __init__(self, in_ch, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super().__init__()
        self.main = Conv2d_LoRA(in_ch, in_ch, 3, stride=1, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, use_cm=True):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x, use_cm)
        return x



class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class ResBlock_CM(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            Conv2d_LoRA(in_ch, out_ch, 3, stride=1, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            Conv2d_LoRA(out_ch, out_ch, 3, stride=1, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode),
        )
        if in_ch != out_ch:
            # self.shortcut = Conv2d_LoRA(in_ch, out_ch, 1, stride=1, padding=0, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode)
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, use_cm=True):
        # h = self.block1(x)
        # print("x shape: ", x.shape)
        # print('block1[0]', self.block1[0])
        # print('block1[0] shape: ', self.block1[0].weight.shape)
        h = self.block1[0](x)
        h = self.block1[1](h)
        h = self.block1[2](h, use_cm)
        h += self.temb_proj(temb)[:, :, None, None]
        # h = self.block2(h)
        h = self.block2[0](h)
        h = self.block2[1](h)
        h = self.block2[2](h)
        h = self.block2[3](h, use_cm)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet_CM(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout,
                 cond, augm, num_class, return_mid=False, r=0, lora_alpha=1,
                 r_ratio=0., scaling=1., lora_mode='value', lora_part=None,
                 coral_projection_dim=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        if cond:
            self.label_embedding = nn.Embedding(num_class, tdim)
        else:
            self.label_embedding = None

        if augm:
            self.augm_embedding = nn.Linear(9, tdim, bias=False)
        else:
            self.augm_embedding = None

        lora_part = tuple(lora_part or ())
        self.lora_part = lora_part
        if "head" not in lora_part:
            self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        else:
            self.head = Conv2d_LoRA(3, ch, kernel_size=3, stride=1, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                if "down" not in lora_part:
                    self.downblocks.append(ResBlock(
                        in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                        dropout=dropout, attn=(i in attn)))
                else:
                    self.downblocks.append(ResBlock_CM(
                        in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                        dropout=dropout, attn=(i in attn), r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode))
                # self.downblocks.append(ResBlock(
                #     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                #     dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                if "down" not in lora_part:
                    self.downblocks.append(DownSample(now_ch))
                else:
                    self.downblocks.append(DownSample_CM(now_ch, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode))
                # self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
        if "middle" not in lora_part:
            self.middleblocks = nn.ModuleList([
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ])
        else:
            self.middleblocks = nn.ModuleList([
                ResBlock_CM(now_ch, now_ch, tdim, dropout, attn=True, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode),
                ResBlock_CM(now_ch, now_ch, tdim, dropout, attn=False, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode),
            ])
        # self.middleblocks = nn.ModuleList([
        #     ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
        #     ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        # ])

        # CORAL's method-specific projection head lives on the common
        # bottleneck.  It is deliberately optional: DDPM/CBDM/T2H/CCUA and
        # IP-SVT must have exactly the plain T2H U-Net state, while CORAL needs
        # the released dense projection before its SupCon loss.  The unused
        # logvar projection is retained because it is part of CORAL's native
        # checkpoint/model contract, even though the published loss uses mean.
        if coral_projection_dim is not None:
            coral_projection_dim = int(coral_projection_dim)
            if coral_projection_dim <= 0:
                raise ValueError('coral_projection_dim must be positive when enabled')
            self.coral_projection_dim = coral_projection_dim
            self.bottleneck_dim = now_ch
            self.mean_proj = nn.Linear(now_ch, coral_projection_dim)
            self.logvar_proj = nn.Linear(now_ch, coral_projection_dim)
        else:
            self.coral_projection_dim = None
            self.mean_proj = None
            self.logvar_proj = None

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                if "up" not in lora_part:
                    self.upblocks.append(ResBlock(
                        in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                        dropout=dropout, attn=(i in attn)))
                else:
                    self.upblocks.append(ResBlock_CM(
                        in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                        dropout=dropout, attn=(i in attn), r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode))
                # self.upblocks.append(ResBlock(
                #     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                #     dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                if "up" not in lora_part:
                    self.upblocks.append(UpSample(now_ch))
                else:
                    self.upblocks.append(UpSample_CM(now_ch, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode))
                # self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        if "tail" not in lora_part:
            self.tail = nn.Sequential(
                nn.GroupNorm(32, now_ch),
                Swish(),
                nn.Conv2d(out_ch, 3, 3, stride=1, padding=1)
            )
        else:
            self.tail = nn.Sequential(
                nn.GroupNorm(32, now_ch),
                Swish(),
                Conv2d_LoRA(out_ch, 3, 3, stride=1, padding=1, r=r, lora_alpha=lora_alpha, r_ratio=r_ratio, scaling=scaling, lora_mode=lora_mode)
            )
        # self.tail = nn.Sequential(
        #     nn.GroupNorm(32, now_ch),
        #     Swish(),
        #     nn.Conv2d(out_ch, 3, 3, stride=1, padding=1)
        # )
        self.return_mid = return_mid
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, y=None, augm=None, use_cm=True, return_mid=None):
        emit_mid = self.return_mid if return_mid is None else bool(return_mid)
        # Timestep embedding
        temb = self.time_embedding(t)

        # Label embedding for conditional generation
        if y is not None and self.label_embedding is not None:
            assert y.shape[0] == x.shape[0]
            temb = temb + self.label_embedding(y)

        # Label embedding for conditional generation
        if augm is not None and self.augm_embedding is not None:
            assert augm.shape[0] == x.shape[0]
            temb = temb + self.augm_embedding(augm)

        # Downsampling
        if "head" not in self.lora_part:
            h = self.head(x)
        else:
            h = self.head(x, use_cm)

        hs = [h]
        if "down" not in self.lora_part:
            for layer in self.downblocks:
                h = layer(h, temb)
                hs.append(h)
        else:
            for layer in self.downblocks:
                h = layer(h, temb, use_cm)
                hs.append(h)
        # for layer in self.downblocks:
        #     h = layer(h, temb)
        #     hs.append(h)
        # Middle
        for layer in self.middleblocks:
            if isinstance(layer, ResBlock_CM):
                h = layer(h, temb, use_cm)
            else:
                h = layer(h, temb)
        temp_mid = h
        if self.coral_projection_dim is not None and emit_mid:
            pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
            feature = self.mean_proj(pooled)
        else:
            feature = temp_mid
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock) or isinstance(layer, ResBlock_CM):
                h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(layer, (ResBlock_CM, UpSample_CM)):
                h = layer(h, temb, use_cm)
            else:
                h = layer(h, temb)
        if "tail" not in self.lora_part:
            h = self.tail(h)
        else:
            h = self.tail[0](h)
            h = self.tail[1](h)
            h = self.tail[2](h, use_cm)

        assert len(hs) == 0
        if emit_mid:
            return h, feature
        else:
            return h
