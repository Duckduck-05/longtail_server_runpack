import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import math

# CM branch can be enabled for selected stages.

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class Conv2d_CM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0,  bias=False, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super(Conv2d_CM, self).__init__()
        # self.r = r
        # self.lora_alpha = lora_alpha
        if lora_mode == 'value':
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        elif lora_mode == 'ratio':
            self.r = max(int(r_ratio * min(in_channels, out_channels)), 1) if r_ratio > 0 else 0
            self.scaling = scaling
        else:
            raise ValueError('mode should be value or ratio')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.r > 0:
            self.lora_A = nn.Parameter(self.conv.weight.new_zeros(self.r*kernel_size, in_channels*kernel_size))
            self.lora_B = nn.Parameter(self.conv.weight.new_zeros(out_channels//self.conv.groups*kernel_size, self.r*kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.r > 0:
            init.kaiming_normal_(self.lora_A, a = math.sqrt(5))
            # init.kaiming_normal_(self.lora_B, a = math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        if self.r > 0:
            if not isinstance(x, list):
                x = [x] * 2
            x_0 = self.conv(x[0])
            weight = self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            x_1 = F.conv2d(x[1], weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            return [x_0, x_1]
        else:
            return self.conv(x) if not isinstance(x, list) else [self.conv(xx) for xx in x]

class Linear_CM(nn.Module):
    def __init__(self, in_features, out_features, bias=True, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super(Linear_CM, self).__init__()
        # self.r = r
        # self.lora_alpha = lora_alpha
        if lora_mode == 'value':
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        elif lora_mode == 'ratio':
            self.r = max(int(r_ratio * min(in_features, out_features)), 1) if r_ratio > 0 else 0
            self.scaling = scaling
        else:
            raise ValueError('mode should be value or ratio')

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.r > 0:
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros(self.r, in_features))
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros(out_features, self.r))
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.r > 0:
            init.kaiming_normal_(self.lora_A, a = math.sqrt(5))
            # init.kaiming_normal_(self.lora_B, a = math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.r > 0:
            x = x if isinstance(x, list) else [x] * 2
            x_0 = self.linear(x[0])
            weight = self.linear.weight + (self.lora_B @ self.lora_A).view(self.linear.weight.shape) * self.scaling
            x_1 = F.linear(x[1], weight, self.linear.bias)
            return [x_0, x_1]
        else:
            return self.linear(x) if not isinstance(x, list) else [self.linear(xx) for xx in x]

        
class NormedLinear_CM(nn.Module):
        def __init__(self, in_features, out_features, r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
            super(NormedLinear_CM, self).__init__()
            # self.r = r
            # self.lora_alpha = lora_alpha
            if lora_mode == 'value':
                self.r = r
                self.lora_alpha = lora_alpha
                self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
            elif lora_mode == 'ratio':
                self.r = max(int(r_ratio * min(in_features, out_features)), 1) if r_ratio > 0 else 0
                self.scaling = scaling
            else:
                raise ValueError('mode should be value or ratio')
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            if self.r > 0:
                self.lora_A = nn.Parameter(self.weight.new_zeros(self.r, in_features))
                self.lora_B = nn.Parameter(self.weight.new_zeros(out_features, self.r))
            self.reset_parameters()
    
        def reset_parameters(self):
            init.kaiming_normal_(self.weight)
            if self.r > 0:
                init.kaiming_normal_(self.lora_A, a = math.sqrt(5))
                # init.kaiming_normal_(self.lora_B, a = math.sqrt(5))
                nn.init.zeros_(self.lora_B)
        
    
        def forward(self, x):
            if self.r > 0:
            #     return F.normalize(x, dim=1).mm(F.normalize(self.weight * (1-self.alpha) + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, dim=0))
            # else:
            #     return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
                x = x if isinstance(x, list) else [x] * 2
                x_0 = F.normalize(x[0], dim=1).mm(F.normalize(self.weight, dim=0))
                x_1 = F.normalize(x[1], dim=1).mm(F.normalize(self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling, dim=0))
                return [x_0, x_1]
            else:
                weight = F.normalize(self.weight, dim=0)
                return F.normalize(x, dim=1).mm(weight) if not isinstance(x, list) else [F.normalize(xx, dim=1).mm(weight) for xx in x]

        

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_CM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', r=0, lora_alpha=1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        super(BasicBlock_CM, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = Conv2d_CM(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, r = r, lora_alpha= lora_alpha, r_ratio= r_ratio, lora_mode = lora_mode, scaling=scaling)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.bn1 = [nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)]
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d_CM(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, r = r, lora_alpha= lora_alpha, r_ratio= r_ratio, lora_mode = lora_mode, scaling=scaling)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.bn2 = [nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)]
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        bs = x.shape[0] if not isinstance(x,list) else x[0].shape[0]
        res = x if isinstance(x,list) else [x,x]
        # res = torch.cat(res, 0)
        out = self.conv1(x)
        # out = torch.cat(out, 0)
        # out = F.relu(self.bn1(out))
        out = [F.relu(self.bn1[0](out[0])), F.relu(self.bn1[1](out[1]))]
        # out = list(torch.split(out, bs, dim=0))
        # out = self.bn2(torch.cat(self.conv2(out), 0))
        out = self.conv2(out)
        out = [self.bn2[0](out[0])+ self.shortcut(res[0]), self.bn2[1](out[1])+ self.shortcut(res[1])]
        # print(out.shape)
        # print(self.shortcut(res).shape)
        # out += self.shortcut(res)
        # out = F.relu(out)
        # out = list(torch.split(out, bs, dim=0))
        out = [F.relu(xx) for xx in out]
        return out
    
class ResNet_s(nn.Module):

    # def __init__(self, block, num_blocks, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, s=30, r=0, lora_alpha = 1,returns_feat=False):
    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, s=30, returns_feat=False, lora_list = [False, False, False, True, False], r = 0, lora_alpha = 1, r_ratio = 0.,scaling=1., lora_mode = 'value'):
        """_summary_

        Args:
            lora_list (list, optional): whether to enable the CM branch for conv1, layer1, layer2, layer3, linear.
            r (int, optional): _description_. Defaults to 0.
            lora_alpha (int, optional): _description_. Defaults to 1.
            r_ratio (_type_, optional): r be the ratio of min(in_c, out_c). Defaults to 0..
            mode (str, optional): 'value' or 'ratio'. 'value' to use r, 'ratio' to use r_ratio
        """
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        self.r = r
        self.lora_alpha = lora_alpha
        self.r_ratio = r_ratio
        self.lora_mode = lora_mode
        self.scaling = scaling

        if lora_list[0]:
            self.conv1 = Conv2d_CM(3, 16, 3, 1, 1, False, r = self.r, lora_alpha= self.lora_alpha, r_ratio = self.r_ratio, lora_mode = self.lora_mode, scaling=self.scaling)
            # self.bn1 = [nn.BatchNorm2d(16), nn.BatchNorm2d(16)]
            self.bn1 = nn.ModuleList([nn.BatchNorm2d(16), nn.BatchNorm2d(16)])
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)

        if lora_list[1]:
            self.layer1 = self._make_layer_cm(BasicBlock_CM, 16, num_blocks[0], stride=1)
        else:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        layer2_output_dim = 32
        layer3_output_dim = 64

        if lora_list[2]:
            self.layer2 = self._make_layer_cm(BasicBlock_CM, layer2_output_dim, num_blocks[1], stride=2)
        else:
            self.layer2 = self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2)
        
        # self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2)
        if lora_list[3]:
            self.layer3 = self._make_layer_cm(BasicBlock_CM, layer3_output_dim, num_blocks[2], stride=2)
        else:
            self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2)

        if use_norm:
            if lora_list[4]:
                self.linear = NormedLinear_CM(layer3_output_dim, num_classes, r = self.r, lora_alpha= self.lora_alpha)
            else:
                self.linear = NormedLinear(layer3_output_dim, num_classes)
        else:
            s = 1
            if lora_list[4]:
                self.linear = Linear_CM(layer3_output_dim, num_classes, r = self.r, lora_alpha= self.lora_alpha)
            else:
                self.linear = nn.Linear(layer3_output_dim, num_classes)
        
        self.s = s
        self.lora_list = lora_list
        self.returns_feat = returns_feat

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def _make_layer_cm(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, r = self.r, lora_alpha = self.lora_alpha, r_ratio = self.r_ratio, lora_mode = self.lora_mode, scaling=self.scaling))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.lora_list[0]:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            bs = x.shape[0]
            # out = self.bn1(torch.cat(self.conv1(x), 0))
            # # out = F.relu(self.bn1(out))
            # out = list(torch.split(out, bs, dim=0))
            out = self.conv1(x)
            out = [F.relu(self.bn1[0](out[0])), F.relu(self.bn1[1](out[1]))]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # self.feat_before_GAP = out
        # out = F.avg_pool2d(out, out.size()[3])
        out = [F.avg_pool2d(m, m.size()[3]) for m in out]
        out = [m.view(m.size(0), -1) for m in out]
        self.feat = out
        # print(out[0].shape)
        # print
        if not self.lora_list[4]:
            out = [self.linear(m)* self.s for m in out]
        else:
            out = self.linear(out)
            out = [self.s * xx for xx in out]
        # out = out * self.s # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
        # final_out = torch.stack(out, dim=1).mean(dim=1)
        if self.returns_feat:
            return {
                "output": out[1], 
                "feat": self.feat,
                "logits": out
            }
        else:
            return {
                "output": out[1],
                "logits": out
            }
    
def create_model(m_type='resnet101',num_classes=1000, use_norm=False, s=30, **kwargs):
    if m_type == 'resnet20':
        return ResNet_s(BasicBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm, s=s, **kwargs)
    elif m_type == 'resnet32':
        return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, s=s, **kwargs)
    elif m_type == 'resnet44':
        return ResNet_s(BasicBlock, [7, 7, 7], num_classes=num_classes, use_norm=use_norm, s=s, **kwargs)
    elif m_type == 'resnet56':
        return ResNet_s(BasicBlock, [9, 9, 9], num_classes=num_classes, use_norm=use_norm, s=s, **kwargs)
    elif m_type == 'resnet110':
        return ResNet_s(BasicBlock, [18, 18, 18], num_classes=num_classes, use_norm=use_norm, s=s, **kwargs)
    else:
        raise ValueError('Network type not supported')
    
