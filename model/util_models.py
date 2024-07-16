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


def clamp_activation(x, ste=False, temp=1.0):
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp - 0.5)) + np.tanh(temp * 0.5)) / (
            2 * (np.tanh(temp * 0.5))
        )
    return (out_s.float() - out_bp).detach() + out_bp


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(bn, x_in, mem, V_th, decay, grad_scale=1.0, temp=1.0):
    mem = mem * decay + x_in
    mem2 = bn(mem)
    spike = clamp_activation(mem2 / V_th, temp=temp)
    mem = mem * (1 - spike)
    return mem, spike


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


class LIFAct(nn.Module):
    """Generates spikes based on LIF module.
    It can be considered as an activation function and is used similar to ReLU.
    The input tensor needs to have an additional time dimension,
    which in this case is on the last dimension of the data."""

    def __init__(self, step, **kwargs):
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 3.0
        self.grad_scale = 0.1
        # if self.grad_scale is None:
        #     self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        self.bn = nn.Identity()

    def forward(self, x):

        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update(
                bn=self.bn,
                x_in=x[i],
                mem=u,
                V_th=self.V_th,
                grad_scale=self.grad_scale,
                decay=0.25,
                temp=self.temp,
            )
            out += [out_i]

        out = torch.stack(out)
        return out


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
