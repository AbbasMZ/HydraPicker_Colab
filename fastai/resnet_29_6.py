import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['ResNet', 'resnet1', 'resnet2', 'resnet3', 'resnet4', 'resnet5', 'resnet6', 'resnet7', 'resnet8', 'resnet11']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1_1 = conv3x3(inplanes, planes, stride)
        self.conv1_2 = conv3x3(planes, planes)
        self.gn1 = GroupNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_1 = conv3x3(planes, planes)
        self.conv2_2 = conv3x3(planes, planes)

        self.gn2 = GroupNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = GroupNorm(64)
        self.relu = nn.ReLU(inplace=True)
        #        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=1)
        # self.layer4 = self._make_layer(block, 64, layers[3], stride=1)
        #        self.avgpool = nn.AvgPool2d(7, stride=1)
        #        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #            elif isinstance(m, nn.BatchNorm2d):
            #                m.weight.data.fill_(1)
            #                m.bias.data.zero_()
            elif isinstance(m, GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #                nn.BatchNorm2d(planes * block.expansion),
                GroupNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gn1(x)
        x = self.relu(x)
        #        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        #        x = self.avgpool(x)
        #        x = x.view(x.size(0), -1)
        #        x = self.fc(x)

        return x

def resnet11(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [5], **kwargs)
    return model

def resnet1(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [1], **kwargs)
    return model

def resnet2(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2], **kwargs)
    return model

def resnet3(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3], **kwargs)
    return model

def resnet4(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [4], **kwargs)
    return model

def resnet5(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [5], **kwargs)
    return model

def resnet6(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [6], **kwargs)
    return model

def resnet7(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [7], **kwargs)
    return model

def resnet8(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [8], **kwargs)
    return model