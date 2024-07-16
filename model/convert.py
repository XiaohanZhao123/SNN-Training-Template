import inspect

import torch
from spikingjelly.activation_based import layer, neuron
from spikingjelly.activation_based.model import sew_resnet
from torch import nn
from .util_models import CostumeLIF, SpikeConv, SpikePool


def ann_to_snn(module: nn.Module, step=2, channel: int = 0):
    """
    Recursively replace the normal conv2d and Linear layer to SpikeLayer in order to accelerate using torch.compile
    """
    for name, child_module in module.named_children():

        if isinstance(child_module, nn.Sequential):
            ann_to_snn(child_module, step=step, channel=channel)

        elif isinstance(child_module, (nn.Conv2d, layer.Conv2d)):
            if isinstance(child_module, layer.Conv2d):
                child_module = spikingjelly_to_ann(child_module, nn.Conv2d)
            setattr(module, name, SpikePool(child_module, step=step))

        elif isinstance(child_module, (nn.Linear, layer.Linear)):
            if isinstance(child_module, layer.Linear):
                child_module = spikingjelly_to_ann(child_module, nn.Linear)

            setattr(module, name, SpikeConv(child_module, step=step))

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
            if isinstance(child_module, layer.AvgPool2d):
                child_module = spikingjelly_to_ann(child_module, nn.AvgPool2d)
            if isinstance(child_module, layer.AdaptiveAvgPool2d):
                child_module = spikingjelly_to_ann(child_module, nn.AdaptiveAvgPool2d)
            if isinstance(child_module, layer.MaxPool2d):
                child_module = spikingjelly_to_ann(child_module, nn.MaxPool2d)

            setattr(module, name, SpikePool(child_module, step=step))

        elif isinstance(
            child_module, (nn.ReLU, nn.ReLU6, neuron.LIFNode, neuron.IFNode)
        ):
            setattr(module, name, CostumeLIF())

        elif isinstance(child_module, (nn.BatchNorm2d, layer.BatchNorm2d)):
            if isinstance(child_module, layer.BatchNorm2d):
                child_module = spikingjelly_to_ann(child_module, nn.BatchNorm2d)
            setattr(module, name, SpikePool(child_module, step=step))
            channel = child_module.num_features

        else:
            ann_to_snn(child_module, step=step, channel=channel)


def spikingjelly_to_ann(spikingjelly_layer, ann_layer):
    """
    Creates an instance of `new_layer_class` using the parameters from `custom_layer`.

    Args:
    - custom_layer (torch.nn.Module): The layer from which to copy parameters.
    - new_layer_class (class): The class of the new layer to be instantiated.

    Returns:
    - torch.nn.Module: New layer instance with copied parameters.
    """
    # Get the constructor signature of the new layer class
    signature = inspect.signature(ann_layer.__init__)
    layer_params = {}

    # Collect parameters that match in the custom_layer
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if hasattr(spikingjelly_layer, name):
            layer_params[name] = getattr(spikingjelly_layer, name)

    if "bias" in layer_params:
        layer_params["bias"] = True if layer_params["bias"] is not None else False

    # Create a new layer instance with the extracted parameters, and copy the weights
    new_layer = ann_layer(**layer_params)
    for name, param in spikingjelly_layer.named_parameters():
        if hasattr(new_layer, name):
            getattr(new_layer, name).data = param.data.clone()

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
