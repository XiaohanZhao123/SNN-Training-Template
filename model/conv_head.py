import torch
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model import sew_resnet
from spikingjelly.activation_based.model.sew_resnet import SEWResNet
from torch import nn

from .util_models import CostumeLIF


def temporal_shuffle(x: torch.Tensor):
    step, batch_size, num_channels, height, width = x.size()
    assert step % 2 == 0
    x = x.view(2, step // 2, batch_size, num_channels, height, width)
    x = x.permute(1, 0, 2, 3, 4, 5)
    x = x.contiguous().view(step, batch_size, num_channels, height, width)
    return x


class FPNHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.block1 = self._get_conv_block(in_channels, 64)
        self.block2 = self._get_conv_block(64, 64)
        self.block3 = self._get_conv_block(64, 64)
        self.pool = layer.AvgPool2d(kernel_size=2, stride=2)

    @staticmethod
    def _get_conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(
            layer.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            layer.BatchNorm2d(out_channels),
            CostumeLIF(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = temporal_shuffle(x)
        x = self.block2(x)
        x = temporal_shuffle(x)
        x = self.block3(x)
        x = temporal_shuffle(x)
        x = self.pool(x)
        return x


def sew_resnet_head18(
    pretrained=False, cnf: str = None, spiking_neuron: callable = None, **kwargs
):
    model: SEWResNet = sew_resnet.sew_resnet18(
        pretrained=pretrained, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs
    )
    model.maxpool = nn.Identity()
    model.bn1 = nn.Identity()
    model.bn1 = nn.Identity()
    input_c = model.conv1.in_channels
    BN = layer.BatchNorm2d

    model.conv1 = nn.Sequential(
        layer.Conv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
        BN(64),
        CostumeLIF(),
        layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        BN(64),
        CostumeLIF(),
        layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        BN(64),
        CostumeLIF(),
        layer.AvgPool2d(2),
    )

    return model


def sew_resnet_fpn18(
    pretrained=False, cnf: str = None, spiking_neuron: callable = None, **kwargs
):
    model: SEWResNet = sew_resnet.sew_resnet18(
        pretrained=pretrained, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs
    )
    model.maxpool = nn.Identity()
    model.bn1 = nn.Identity()
    model.bn1 = nn.Identity()
    input_c = model.conv1.in_channels
    BN = layer.BatchNorm2d
    model.conv1 = FPNHead(input_c)
    return model
