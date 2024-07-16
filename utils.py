from typing import Type

import torch
from omegaconf import DictConfig, OmegaConf
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model.sew_resnet import (sew_resnet18,
                                                            sew_resnet34,
                                                            sew_resnet50,
                                                            sew_resnet101,
                                                            sew_resnet152)
from spikingjelly.activation_based.model.spiking_resnet import (
    spiking_resnet18, spiking_resnet34, spiking_resnet50, spiking_resnet101,
    spiking_resnet152)
from spikingjelly.activation_based.model.spiking_vgg import (spiking_vgg11_bn,
                                                             spiking_vgg13_bn,
                                                             spiking_vgg16_bn,
                                                             spiking_vgg19_bn)
from spikingjelly.activation_based.model.tv_ref_classify import transforms
from torch import Dict, nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from data.auto_argument import CIFAR10Policy, Cutout
from model.conv_head import sew_resnet_fpn18, sew_resnet_head18
from model.convert import ann_to_snn
from model.resnet import SpikeModel, resnet20_cifar_modified
from model.temporal_inflation import (TemporalInflationNeuron,
                                      TemporalInflationWeightedSumNeuron,
                                      TemproalInflationAdd)
from model.util_models import LIFAct, clamp_activation, myBatchNorm3d

model_dict: Dict[str, Type[nn.Module]] = {
    "sew_resnet18": sew_resnet18,
    "sew_resnet34": sew_resnet34,
    "sew_resnet50": sew_resnet50,
    "sew_resnet101": sew_resnet101,
    "sew_resnet152": sew_resnet152,
    "spiking_vgg11_bn": spiking_vgg11_bn,
    "spiking_vgg13_bn": spiking_vgg13_bn,
    "spiking_vgg16_bn": spiking_vgg16_bn,
    "spiking_vgg19_bn": spiking_vgg19_bn,
    "spiking_resnet18": spiking_resnet18,
    "spiking_resnet34": spiking_resnet34,
    "spiking_resnet50": spiking_resnet50,
    "spiking_resnet101": spiking_resnet101,
    "spiking_resnet152": spiking_resnet152,
    "sew_head": sew_resnet_head18,
    "sew_fpn": sew_resnet_fpn18,
}

new_model_dict: Dict[str, Type[nn.Module]] = {
    "resnet20_cifar_modified": resnet20_cifar_modified,
}


loss_dict: Dict[str, nn.Module] = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
}

inflation_dict: Dict[str, Type[nn.Module]] = {
    "binary": TemporalInflationNeuron,
    "conv": TemporalInflationWeightedSumNeuron,
    "add": TemproalInflationAdd,
}


def get_model(cfg: DictConfig):
    model_cls = model_dict.get(cfg.model.name)
    model_cls = new_model_dict.get(cfg.model.name) if model_cls is None else model_cls
    assert model_cls is not None, f"Model {cfg.model.name} not found in model_dict"
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    del model_kwargs["name"]
    del model_kwargs["path"]
    spiking_neuron = neuron.LIFNode
    surrogate_fn = surrogate.ATan()
    if cfg.model.name in model_dict:

        model = model_cls(
            spiking_neuron=spiking_neuron,
            surrogate_function=surrogate_fn,
            **model_kwargs,
        )

        if cfg.replace_activation is True:
            replace_activation_into_temporal_inflation(
                model,
                neuron.LIFNode,
                inflation_dict[cfg.inflation_type],
                p=cfg.p,
                neuron_cls=spiking_neuron,
                surrogate_function=surrogate_fn,
            )

        if cfg.dataset.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)

        functional.set_step_mode(model, "m")
        functional.set_backend(model, "torch")

        if cfg.to_pytorch is True:
            ann_to_snn(model, step=cfg.T)

        return model

    else:
        model = model_cls(**model_kwargs)
        model = SpikeModel(model=model, step=cfg.T)
        return model


def get_dataset(cfg: DictConfig):
    """
    Get the dataset for training and testing.

    Args:
        cfg (DictConfig): Configuration dictionary containing dataset information.

    Returns:
        tuple: A tuple containing the dataset class and a dictionary of dataset arguments for training and testing.
    """
    # use basic argumentation, need to add more advanced argumentation in the future
    augmentation_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.data.auto_argu is True:
        augmentation_list.append(CIFAR10Policy())
    augmentation_list.append(transforms.ToTensor())

    if cfg.data.cutout is True:
        augmentation_list.append(Cutout(n_holes=1, length=16))

    augmentation_list.append(
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    )
    train_transform = transforms.Compose(augmentation_list)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std),
        ]
    )

    dataset_cls = globals().get(cfg.dataset.name)
    assert dataset_cls is not None, f"Dataset {cfg.dataset.name} not implemented"

    train_kwargs = {"transform": train_transform, "root": cfg.dataset.root}
    test_kwargs = {"transform": test_transform, "root": cfg.dataset.root}

    return dataset_cls, {"train": train_kwargs, "test": test_kwargs}


def get_loss(cfg: DictConfig):
    loss_fn = loss_dict.get(cfg.loss)
    assert loss_fn is not None, f"Loss function {cfg.loss} not found in loss_dict"

    return loss_fn


def replace_activation(
    model: nn.Module,
    activation_cls: Type[nn.Module],
    custom_activation_cls: Type[nn.Module],
    **kwargs,
):
    """
    Replaces the specified activation function in a PyTorch model with a custom activation function.

    Args:
        model (nn.Module): The PyTorch model in which to replace the activation function.
        activation_cls (Type[nn.Module]): The type of activation function to replace.
        custom_activation_cls (Type[nn.Module]): The type of custom activation function to replace with.
        **kwargs: Additional keyword arguments to pass to the custom activation function.

    Returns:
        None
    """
    for name, module in model.named_children():
        if isinstance(module, activation_cls):
            setattr(model, name, custom_activation_cls(**kwargs))
        else:
            replace_activation(module, activation_cls, custom_activation_cls, **kwargs)


def replace_activation_into_temporal_inflation(
    module: nn.Module,
    activation_cls: Type[nn.Module],
    custom_activation_cls: Type[nn.Module],
    **kwargs,
):
    """
    Replaces the specified activation function in a PyTorch module with a custom activation function.

    Args:
        module (nn.Module): The PyTorch module in which to replace the activation function.
        activation_cls (Type[nn.Module]): The type of activation function to replace.
        custom_activation_cls (Type[nn.Module]): The type of custom activation function to replace with.
        **kwargs: Additional keyword arguments to pass to the custom activation function.

    Returns:
        None
    """
    for name, sub_module in module.named_children():
        if isinstance(
            sub_module, activation_cls
        ):  # Change this to the type of activation you want to replace
            # Get the previous layer's output channels
            prev_layer = list(module.children())[
                list(module.children()).index(sub_module) - 1
            ]
            in_channels = _get_output_channels(prev_layer)
            # Replace the activation function with the custom one
            setattr(module, name, custom_activation_cls(in_channels, **kwargs))
        else:
            replace_activation_into_temporal_inflation(
                sub_module, activation_cls, custom_activation_cls, **kwargs
            )  # Recursively apply to child modules


def replace_bn(module: nn.Module):
    """
    Replaces the specified activation function in a PyTorch module with a custom activation function.

    Args:
        module (nn.Module): The PyTorch module in which to replace the activation function.
        activation_cls (Type[nn.Module]): The type of activation function to replace.
        custom_activation_cls (Type[nn.Module]): The type of custom activation function to replace with.
        **kwargs: Additional keyword arguments to pass to the custom activation function.

    Returns:
        None
    """
    for name, sub_module in module.named_children():
        if isinstance(
            sub_module, nn.BatchNorm2d
        ):  # Change this to the type of activation you want to replace
            setattr(module, name, myBatchNorm3d(sub_module))
        else:
            replace_bn(
                sub_module,
            )


# Helper function to get the output channels of the last layer in a block
def _get_output_channels(module: nn.Module):
    """
    Get the number of output channels of a given module.

    Args:
        module (nn.Module): The module for which to get the output channels.

    Returns:
        int: The number of output channels.

    Raises:
        ValueError: If the module type is not supported.
    """
    if isinstance(module, (nn.Conv2d, layer.Conv2d)):
        return module.out_channels
    elif isinstance(module, (nn.Linear, layer.Linear)):
        return module.out_features
    elif isinstance(module, (nn.BatchNorm2d, layer.BatchNorm2d)):
        return module.num_features
    elif hasattr(module, "children") and list(module.children()):
        # Recursively get the output channels of the last child
        return _get_output_channels(list(module.children())[-1])
    else:
        raise ValueError(f"Unsupported module type: {type(module)}")
