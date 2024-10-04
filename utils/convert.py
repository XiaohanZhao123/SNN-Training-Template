import inspect
from typing import Type, Union

import torch
from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.model import sew_resnet
from torch import nn

from .util_models import CustomLIF, TemporalConvWrapper, TemporalLinearWrapper


def ann_to_snn(module: nn.Module, step: int = 2):
    """
    Recursively replace the normal conv2d and Linear layer to SpikeLayer to accelerate using torch.compile
    """
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Sequential):
            ann_to_snn(child_module, step=step)
        elif isinstance(
            child_module, (nn.Conv2d, nn.Linear, layer.Conv2d, layer.Linear)
        ):
            new_module = convert_to_temporal_wrapper(child_module, step)
            setattr(module, name, new_module)
        elif isinstance(
            child_module,
            (
                nn.AdaptiveAvgPool2d,
                nn.AvgPool2d,
                nn.MaxPool2d,
                layer.AdaptiveAvgPool2d,
                layer.AvgPool2d,
                layer.MaxPool2d,
            ),
        ):
            new_module = convert_to_temporal_wrapper(child_module, step)
            setattr(module, name, new_module)
        elif isinstance(
            child_module, (nn.ReLU, nn.ReLU6, neuron.LIFNode, neuron.IFNode)
        ):
            setattr(module, name, CustomLIF())
        elif isinstance(child_module, (nn.BatchNorm2d, layer.BatchNorm2d)):
            new_module = convert_to_temporal_wrapper(child_module, step)
            setattr(module, name, new_module)
        else:
            ann_to_snn(child_module, step=step)


def convert_to_temporal_wrapper(module: nn.Module, step: int) -> TemporalConvWrapper:
    """
    Convert a module to its temporal wrapper version.
    """
    if isinstance(
        module,
        (
            layer.Conv2d,
            layer.Linear,
            layer.AvgPool2d,
            layer.AdaptiveAvgPool2d,
            layer.MaxPool2d,
            layer.BatchNorm2d,
        ),
    ):
        module = spikingjelly_to_ann(module)

    if isinstance(module, nn.Linear):
        return TemporalLinearWrapper(module, step=step)

    return TemporalConvWrapper(module, step=step)


def spikingjelly_to_ann(spikingjelly_layer: nn.Module) -> nn.Module:
    """
    Convert a SpikingJelly layer to its PyTorch equivalent.
    """
    ann_layer_mapping = {
        layer.Conv2d: nn.Conv2d,
        layer.Linear: nn.Linear,
        layer.AvgPool2d: nn.AvgPool2d,
        layer.AdaptiveAvgPool2d: nn.AdaptiveAvgPool2d,
        layer.MaxPool2d: nn.MaxPool2d,
        layer.BatchNorm2d: nn.BatchNorm2d,
    }

    ann_layer_class = ann_layer_mapping.get(type(spikingjelly_layer))
    if not ann_layer_class:
        raise ValueError(f"Unsupported layer type: {type(spikingjelly_layer)}")

    return create_ann_layer(spikingjelly_layer, ann_layer_class)


def create_ann_layer(
    custom_layer: nn.Module, new_layer_class: Type[nn.Module]
) -> nn.Module:
    """
    Create an instance of `new_layer_class` using the parameters from `custom_layer`.
    """
    signature = inspect.signature(new_layer_class.__init__)
    layer_params = {
        name: getattr(custom_layer, name)
        for name, param in signature.parameters.items()
        if name != "self" and hasattr(custom_layer, name)
    }

    if "bias" in layer_params:
        layer_params["bias"] = layer_params["bias"] is not None

    new_layer = new_layer_class(**layer_params)

    # Copy weights
    for name, param in custom_layer.named_parameters():
        if hasattr(new_layer, name):
            getattr(new_layer, name).data.copy_(param.data)

    return new_layer


if __name__ == "__main__":
    model = sew_resnet.sew_resnet18(
        cnf="ADD", spiking_neuron=neuron.LIFNode, pretrained=True
    )
    ann_to_snn(model)
    print(model)
    x = torch.randn(10, 1, 3, 32, 32)
    out = model(x)
    print(out)
