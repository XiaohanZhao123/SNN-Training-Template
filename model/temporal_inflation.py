"""
Implementation of the Temporal Inflation model.
"""

from typing import Mapping, Type

import torch
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch import nn


class TemporalInflationNeuron(nn.Module):

    def __init__(
        self, in_channels: int, p: int, neuron_cls: Type[neuron.BaseNode], **kwargs
    ):
        super().__init__()
        self.p = p
        self.act1 = neuron_cls(**kwargs)
        self.temproal_fusion_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(p, 1, 1),
            stride=(p, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.act2 = neuron_cls(**kwargs)

    def forward(self, x: torch.Tensor):
        # repeat x along dim 0
        x = torch.repeat_interleave(x, self.p, dim=0)
        x = self.act1(x)
        x = x.permute(1, 2, 0, 3, 4)
        x = self.temproal_fusion_layer(x)
        x = x.permute(2, 0, 1, 3, 4)
        return self.act2(x)


class TemporalInflationWeightedSumNeuron(nn.Module):

    def __init__(
        self, in_channels, p: int, neuron_cls: Type[neuron.BaseNode], **kwargs
    ):
        super().__init__()
        self.p = p
        self.act1 = neuron_cls(**kwargs)
        self.temproal_fusion_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(p, 1, 1),
            stride=(p, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        # repeat x along dim 0
        x = x.repeat(self.p, 1, 1, 1, 1)
        x = self.act1(x)
        x = x.permute(1, 2, 0, 3, 4)
        x = self.temproal_fusion_layer(x)
        x = x.permute(2, 0, 1, 3, 4)
        return x


class TemproalInflationAdd(nn.Module):

    def __init__(
        self, in_channels: int, p: int, neuron_cls: Type[neuron.BaseNode], **kwargs
    ):
        super().__init__()
        self.p = p
        self.act1 = neuron_cls(**kwargs)

    def forward(self, x: torch.Tensor):
        # repeat x along dim 0
        x = x.repeat(self.p, 1, 1, 1, 1)
        x = self.act1(x)
        # split x into p parts in dim 0 then add them
        xs = torch.chunk(x, self.p, dim=0)
        x = torch.sum(torch.stack(xs), dim=0)
        return x


if __name__ == '__main__':
    # Create an instance of TemproalInflationAdd
    in_channels = 3
    p = 4
    neuron_cls = neuron.LIFNode
    kwargs = {'v_threshold': 1.0, 'v_reset': 0.0}
    model = TemproalInflationAdd(in_channels, p, neuron_cls, **kwargs)

    # Create input tensor
    batch_size = 2
    height = 5
    width = 5
    input_tensor = torch.randn(10, batch_size, in_channels, height, width)

    # Forward pass
    output_tensor = model(input_tensor)

    # Check output shape
    print(output_tensor.shape)

