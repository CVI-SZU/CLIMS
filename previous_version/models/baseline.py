import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.leaky_relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.leaky_relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)


class Baseline(nn.Module):
    def __init__(self, num_classes=20, arch='resnet50'):
        super(Baseline, self).__init__()

        if arch == 'resnet18':
            model = resnet18(pretrained=True, num_classes=num_classes, stride=[1, 2, 2, 1])
        elif arch == 'resnet34':
            model = resnet34(pretrained=True, num_classes=num_classes, stride=[1, 2, 2, 1])
        elif arch == 'resnet50':
            model = resnet50(pretrained=True, num_classes=num_classes, stride=[1, 2, 2, 1])
        elif arch == 'resnet101':
            model = resnet101(pretrained=True, num_classes=num_classes, stride=[1, 2, 2, 1])
        elif arch == 'resnet152':
            model = resnet152(pretrained=True, num_classes=num_classes, stride=[1, 2, 2, 1])
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(2048, num_classes, kernel_size=1, padding=0, bias=False)

        self._xavier_init(self.fc)

        self.feat_dim = 2048
        self.from_scratch_layers = [self.fc]

    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _xavier_init(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x, with_cams=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # [N, 256, 112, 112]
        x_512 = self.layer2(x)
        x_1024 = self.layer3(x_512)
        x_2048 = self.layer4(x_1024)
        if with_cams:
            cams = self.fc(x_2048)
            logits = self.avg(cams).reshape(x.size(0), -1)

            return x_2048, cams
        else:
            feats = self.avg(x_2048)
            logits = self.fc(feats).reshape(x.size(0), -1)

            return logits

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


if __name__ == '__main__':
    model = HRCAM()
    # model.get_parameter_groups()
    print(model)
    inp = torch.randn(2, 3, 448, 448)
    label = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 1]])
    out = model(inp)
