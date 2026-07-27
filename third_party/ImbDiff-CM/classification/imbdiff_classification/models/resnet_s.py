import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

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


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, s=30):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2 = self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2)

        if use_norm:
            self.linear = NormedLinear(layer3_output_dim, num_classes)
        else:
            s = 1
            self.linear = nn.Linear(layer3_output_dim, num_classes)
        
        self.s = s

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.feat_before_GAP = out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.feat = out
        
        out = self.linear(out)
        out = out * self.s # This hyperparam s is originally in the loss function, but we moved it here to prevent using s multiple times in distillation.
        return {
            'output': out
        }
    
def create_model(m_type='resnet101',num_classes=1000, use_norm=False, reduce_dimension=False, layer3_output_dim=None, layer2_output_dim=None, s=30):
    if m_type == 'resnet20':
        return ResNet_s(BasicBlock, [3, 3, 3], num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer2_output_dim=layer2_output_dim, s=s)
    elif m_type == 'resnet32':
        return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer2_output_dim=layer2_output_dim, s=s)
    elif m_type == 'resnet44':
        return ResNet_s(BasicBlock, [7, 7, 7], num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer2_output_dim=layer2_output_dim, s=s)
    elif m_type == 'resnet56':
        return ResNet_s(BasicBlock, [9, 9, 9], num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer2_output_dim=layer2_output_dim, s=s)
    elif m_type == 'resnet110':
        return ResNet_s(BasicBlock, [18, 18, 18], num_classes=num_classes, use_norm=use_norm, reduce_dimension=reduce_dimension, layer3_output_dim=layer3_output_dim, layer2_output_dim=layer2_output_dim, s=s)
    else:
        raise ValueError('Network type not supported')
    