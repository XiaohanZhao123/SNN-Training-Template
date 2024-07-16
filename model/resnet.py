"""
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util_models import LIFAct, SpikeConv, SpikePool, myBatchNorm3d


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = ReLU(inplace=True)

    def forward(self, s):
        temp, x = s
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)

        out = self.conv2(out)

        out = self.bn2(out)
        # print(out.min())
        # print(out.max())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out1 = self.relu2(out.clone())
        # print(out.min())
        # print(out1.min())

        return out, out1


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BN(planes)
        self.relu2 = ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BN(planes * self.expansion)
        self.relu3 = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=10):
        super(ResNet_Cifar, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU

        # self.scale = nn.Parameter(torch.ones(1, 1, 2048), requires_grad=True)
        # self.rp = rp
        inplanes = 128
        self.inplanes = 128
        self.conv1 = nn.Conv2d(
            input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 4 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BN(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        if not feat is None:
            x = self.fc(feat)
            return x

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            temp, x = self.layer1((x, x))
            temp, x = self.layer2((temp, x))
            temp, x = self.layer3((temp, x))
            if is_drop:
                temp = F.relu(temp)
                x = self.avgpool(temp)
            else:
                x = self.avgpool(x)

            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)

                fea = x.mean([0])
            x = self.fc(x)
            if is_adain:
                return fea, x
            else:
                return x

    def print_t(self):
        for n, p in self.named_parameters():
            if "threshold" in n:
                print("Param {}, Value {}".format(n, p.data.item()))


class ResNet(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=1000):
        super(ResNet, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            input_c, inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BN(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        if not feat is None:
            x = self.fc(feat)
            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool1(x)

            temp, x = self.layer1((x, x))
            temp, x = self.layer2((temp, x))
            temp, x = self.layer3((temp, x))
            temp, x = self.layer4((temp, x))

            if is_drop:
                temp = F.relu(temp)
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)

            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x = self.fc(x)

            if is_adain:
                return fea, x
            else:
                return x

    def print_t(self):
        for n, p in self.named_parameters():
            if "threshold" in n:
                print("Param {}, Value {}".format(n, p.data.item()))


class ResNet_Cifar_Modified(nn.Module):

    def __init__(self, block, layers, num_classes=10, input_c=3, rp=False):
        super(ResNet_Cifar_Modified, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU

        # self.scale = nn.Parameter(torch.ones(1, 1, 512), requires_grad=True)
        self.rp = rp

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

        # zero_init_residual:
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # AvgDown Layer
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    stride, stride=stride, ceil_mode=True, count_include_pad=False
                ),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                BN(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):

        if not feat is None:
            x = self.fc(feat)
            return x
        else:
            x = self.conv1(x)
            x = self.avgpool(x)
            temp, x = self.layer1((x, x))
            temp, x = self.layer2((temp, x))
            temp, x = self.layer3((temp, x))
            temp, x = self.layer4((temp, x))
            # print(temp.sum())
            # print(x.sum())
            if is_drop:
                temp = F.relu(temp)
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)

            x = x.view(x.size(0), x.size(1), -1)
            x = self.fc(x)

            return x


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet20_cifar_modified(**kwargs):
    model = ResNet_Cifar_Modified(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet19_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 2], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, step=2):
        super().__init__()
        self.model = model
        self.step = step
        self.spike_module_refactor(self.model, step=step)
        self.channel = 0

    def spike_module_refactor(self, module: nn.Module, step=2):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():

            if isinstance(child_module, nn.Sequential):
                self.spike_module_refactor(child_module, step=step)

            elif isinstance(child_module, nn.Conv2d):
                setattr(module, name, SpikePool(child_module, step=step))

            elif isinstance(child_module, nn.Linear):
                setattr(module, name, SpikeConv(child_module, step=step))

            elif isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                setattr(module, name, SpikePool(child_module, step=step))

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                setattr(module, name, LIFAct(step=step, channel=self.channel))
            # elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, SpikeConv(child_module, step=step))
            # elif isinstance(child_module, nn.BatchNorm2d):
            #    setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))
            elif isinstance(child_module, nn.BatchNorm2d):
                setattr(module, name, myBatchNorm3d(child_module, step=step))
                # elif isinstance(child_module, nn.BatchNorm2d):
                #    setattr(module, name, myNone( step=step))

                self.channel = child_module.num_features

            else:
                self.spike_module_refactor(child_module, step=step)

    def forward(self, input, is_adain=False, is_drop=False):

        if is_adain and is_drop:
            fea, out = self.model(input, is_adain=True, is_drop=True)
        elif is_adain and not is_drop:
            fea, out = self.model(input, is_adain=True, is_drop=False)
        elif not is_adain and is_drop:
            out = self.model(input, is_adain=False, is_drop=True)
        else:
            out = self.model(input, is_adain=False, is_drop=False)
        if is_adain:
            return fea, out
        else:
            return out


# from models.resnet import resnet20_cifar_modified
# model = SpikeModel(resnet20_cifar_modified())
# model.set_spike_before('layer1')
# for n, m in model.named_modules():
#     if isinstance(m, SpikeModule):
#         if m._spiking is True:
#             print(n)
# import torch
# model(torch.randn(1,3,32,32))


if __name__ == "__main__":
    net = resnet20_cifar()
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y1 = net(x)
