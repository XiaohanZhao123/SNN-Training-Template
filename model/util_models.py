import math
from typing import Callable

import numpy as np
import torch
from spikingjelly.activation_based import layer, model, neuron, surrogate
from spikingjelly.activation_based.model import sew_resnet
from torch import nn


class myBatchNorm3d(nn.Module):
    def __init__(self, BN: nn.BatchNorm2d, step=2):
        super().__init__()
        self.bn = nn.BatchNorm3d(BN.num_features)
        self.step = step

    def forward(self, x):
        # T N C H W -> N C T H W
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def _atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None


def heaviside(x):
    return (x > 0).to(x)


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return _atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class CostumeLIF(nn.Module):
    def __init__(
        self,
        tau: float = 2,
        v_threshold: float = 1.0,
        surrgate_function: Callable = atan,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.surrgate_function = surrgate_function

    def forward(self, x):
        v = torch.zeros_like(x[0])
        spike_outs = []
        for x_input in x:
            v = v + (x_input - v) / self.tau
            spike = self.surrgate_function.apply(v - self.v_threshold, 2.0)
            v = v * (1 - spike)
            spike_outs.append(spike)

        return torch.stack(spike_outs)


class SpikeConv(nn.Module):

    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step
        self.step_mode = "m"

    def forward(self, x):
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out


class SpikePool(nn.Module):

    def __init__(self, pool, step=2):
        super().__init__()
        self.pool = pool
        self.step = step
        self.step_mode = "m"

    def forward(self, x):
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out
