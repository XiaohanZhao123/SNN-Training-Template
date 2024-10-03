"""
This file contains functions to instantiate models, datasets, and other components.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import (sew_resnet, spiking_resnet,
                                                 spiking_vgg,
                                                 spiking_vggws_ottt)
from spikingjelly.datasets import cifar10_dvs, dvs128_gesture, n_mnist
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from data.kornia_augmentation import (get_neuromorphic_dataset_augmentation,
                                      get_vision_dataset_augmentation)
from data.lmdb import LMDBDataset
from model.convert import ann_to_snn

MODEL_DICT = {
    "sew_resnet": sew_resnet.sew_resnet34,
    "spiking_vgg": spiking_vgg.spiking_vgg16,
    "spiking_resnet": spiking_resnet.spiking_resnet18,
    "spiking_vggws_ottt": spiking_vggws_ottt.ottt_spiking_vgg16_ws,
    "sew_resnet_modified": sew_resnet.sew_resnet34,
}

NEUROMORPHIC_DATASET_DICT = {
    "dvs128_gesture": dvs128_gesture,
    "cifar10_dvs": cifar10_dvs,
    "n_mnist": n_mnist,
}

TORCHVISION_DATASET_DICT = {"cifar10": CIFAR10, "mnist": MNIST, "cifar100": CIFAR100}

CONFIG = {"default_frames_number": 16, "train_split_ratio": 0.8}


def get_model(
    model_name: str, num_classes: int, to_pytorch: bool, T: int, **model_kwargs
):
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model: {model_name}")
    spiking_neuron = neuron.LIFNode
    spiking_neuron_kwargs = {
        "tau": 2.0,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "surrogate_function": surrogate.ATan(),
        "step_mode": "m",
    }
    # merge model_kwargs and spiking_neuron_kwargs
    model_kwargs = {**model_kwargs, **spiking_neuron_kwargs}
    model = MODEL_DICT[model_name](**model_kwargs, spiking_neuron=spiking_neuron)
    # change the output layer to match the number of classes
    model.fc = layer.Linear(model.fc.in_features, num_classes)
    if model_name == "sew_resnet_modified":
        print('using modified model, conv1 will be replaced')
        model.conv1 = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.BatchNorm2d(64, step_mode="m"),
            neuron.LIFNode(**spiking_neuron_kwargs),
            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.BatchNorm2d(64, step_mode="m"),
            neuron.LIFNode(**spiking_neuron_kwargs),
            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.BatchNorm2d(64, step_mode="m"),
            neuron.LIFNode(**spiking_neuron_kwargs),
        )
    functional.set_step_mode(model, "m")
    functional.set_backend(model, "torch")
    
    if to_pytorch:
        ann_to_snn(model, step=T)
    else:
        functional.set_backend(model, "cupy")
    return model


def get_transform(dataset_name, use_lmdb):
    if dataset_name.lower() in NEUROMORPHIC_DATASET_DICT:
        if use_lmdb:
            return None, None
        else:
            # from numpy since to tensor only works for torchvision datasets
            return torch.from_numpy, torch.from_numpy
    else:
        return transforms.ToTensor(), transforms.ToTensor()


def get_augmentation(dataset_name, img_size, mean, std, autoaugment):
    if dataset_name.lower() in NEUROMORPHIC_DATASET_DICT:
        return get_neuromorphic_dataset_augmentation()
    else:
        return get_vision_dataset_augmentation(img_size, mean, std, autoaugment)


def get_dataset(
    dataset_name: str,
    root: str,
    img_size: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    frame_number: Optional[int] = None,
    use_lmdb: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the appropriate dataset based on the dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        root (str): Root directory for the dataset.
        img_size (int): Size of the image.
        mean (Tuple[float, ...]): Mean values for normalization.
        std (Tuple[float, ...]): Standard deviation values for normalization.
        frame_number (int, optional): Number of frames to use for neuromorphic datasets.
        use_lmdb (bool, optional): Whether to use LMDB to accelerate data loading.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.
    """
    train_transform, test_transform = get_transform(dataset_name, use_lmdb)

    if dataset_name.lower() in NEUROMORPHIC_DATASET_DICT:
        return _get_neuromorphic_dataset(
            dataset_name,
            root,
            CONFIG["default_frames_number"] if frame_number is None else frame_number,
            train_transform,
            test_transform,
            use_lmdb=use_lmdb,
        )
    else:
        return _get_torchvision_dataset(
            dataset_name, root, train_transform, test_transform
        )


@lru_cache(maxsize=None)
def _check_lmdb_path(path: str) -> bool:
    """
    Check if the given path contains valid LMDB files for both train and val sets.

    Args:
        path (str): Path to the LMDB directory.

    Returns:
        bool: True if valid LMDB files exist, False otherwise.
    """
    path = Path(path)
    # check if subdirectories "train" and "val" exist
    if not all((path / subset).is_dir() for subset in ["train", "val"]):
        return False
    return all(
        any(file.suffix == ".mdb" for file in (path / subset).iterdir())
        for subset in ["train", "val"]
    )


def _get_neuromorphic_dataset(
    dataset_name: str,
    root: str,
    frames_number: int = CONFIG["default_frames_number"],
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    use_lmdb: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get a neuromorphic dataset.

    Args:
        dataset_name (str): Name of the dataset.
        root (str): Root directory for the dataset.
        frames_number (int): Number of frames to use.
        train_transform (transforms.Compose, optional): Transform for training data.
        test_transform (transforms.Compose, optional): Transform for testing data.
        use_lmdb (bool, optional): Whether to use LMDB to accelerate data loading.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    dataset_class = NEUROMORPHIC_DATASET_DICT.get(dataset_name.lower())
    if not dataset_class:
        raise ValueError(f"Unknown neuromorphic dataset: {dataset_name}")

    lmdb_path = f"./resources/datasets/lmdb/{dataset_name}"

    if use_lmdb:
        if _check_lmdb_path(lmdb_path):
            return _get_lmdb_datasets(lmdb_path, train_transform, test_transform)
        else:
            print(
                f"LMDB dataset not found at {lmdb_path}, falling back to regular dataset."
            )

    return _get_regular_neuro_datasets(
        dataset_class,
        root,
        frames_number,
        train_transform,
        test_transform,
        dataset_name,
    )


def _get_lmdb_datasets(
    lmdb_path: str,
    train_transform: Optional[transforms.Compose],
    test_transform: Optional[transforms.Compose],
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get train, validation, and test datasets from LMDB.

    Args:
        lmdb_path (str): Path to the LMDB directory.
        train_transform (transforms.Compose, optional): Transform for training data.
        test_transform (transforms.Compose, optional): Transform for testing data.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.

    Raises:
        FileNotFoundError: If the LMDB directories are not found.
        RuntimeError: If there's an error creating the LMDBDataset.
    """
    train_path = Path(lmdb_path) / "train"
    test_path = Path(lmdb_path) / "val"  # Use 'val' as test set

    if not all(path.exists() for path in [train_path, test_path]):
        raise FileNotFoundError(f"LMDB directories not found at {lmdb_path}")

    try:
        train_dataset = LMDBDataset(
            lmdb_path=str(train_path), transform=train_transform
        )
        test_dataset = LMDBDataset(lmdb_path=str(test_path), transform=test_transform)
    except Exception as e:
        raise RuntimeError(f"Error creating LMDBDataset: {str(e)}")

    if any(len(dataset) == 0 for dataset in [train_dataset, test_dataset]):
        raise ValueError("One or more datasets are empty")

    # Split train into train and validation
    train_dataset, val_dataset = _split_train_val(
        train_dataset, val_transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def _get_regular_neuro_datasets(
    dataset_class,
    root: str,
    frames_number: int,
    train_transform: Optional[transforms.Compose],
    test_transform: Optional[transforms.Compose],
    dataset_name: str,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get train, validation, and test datasets for regular (non-LMDB) datasets.

    Args:
        dataset_class: The dataset class to instantiate.
        root (str): Root directory for the dataset.
        frames_number (int): Number of frames to use.
        train_transform (transforms.Compose, optional): Transform for training data.
        test_transform (transforms.Compose, optional): Transform for testing data.
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.
    """
    common_args = {
        "root": root,
        "data_type": "frame",
        "frames_number": frames_number,
        "split_by": "number",
    }

    if dataset_name.lower() in ["dvs-cifar10", "n-caltech101"]:
        dataset = dataset_class(**common_args)
        train_dataset, test_dataset = _split_dataset(
            dataset, train_transform, test_transform
        )
        train_dataset, val_dataset = _split_train_val(train_dataset, test_transform)
        return train_dataset, val_dataset, test_dataset
    else:
        train_dataset = dataset_class(
            **common_args, transform=train_transform, train=True
        )
        test_dataset = dataset_class(
            **common_args, transform=test_transform, train=False
        )
        train_dataset, val_dataset = _split_train_val(train_dataset, test_transform)
        return train_dataset, val_dataset, test_dataset


def _get_torchvision_dataset(
    dataset_name: str,
    root: str,
    train_transform: transforms.Compose = None,
    test_transform: transforms.Compose = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get a torchvision dataset.

    Args:
        dataset_name (str): Name of the dataset.
        root (str): Root directory for the dataset.
        train_transform (transforms.Compose, optional): Transform for training data.
        test_transform (transforms.Compose, optional): Transform for testing data.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets.
    """
    dataset_class = TORCHVISION_DATASET_DICT.get(dataset_name.lower())
    if not dataset_class:
        raise ValueError(f"Unknown torchvision dataset: {dataset_name}")

    train_dataset = dataset_class(
        root=root, train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_class(
        root=root, train=False, download=True, transform=test_transform
    )

    train_dataset, val_dataset = _split_train_val(train_dataset, test_transform)

    return train_dataset, val_dataset, test_dataset


def _split_train_val(
    dataset: Dataset, val_transform: transforms.Compose
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation sets.

    Args:
        dataset (Dataset): The dataset to split.

    Returns:
        Tuple[Dataset, Dataset]: Train and validation datasets.
    """
    val_size = int((1 - CONFIG["train_split_ratio"]) * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset = deepcopy(
        val_dataset
    )  # deepcopy is necessary since random_split returns a Subset object
    val_dataset.transform = val_transform

    return train_dataset, val_dataset


def _split_dataset(
    dataset: Dataset,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into training and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_transform (transforms.Compose): Transform for training data.
        test_transform (transforms.Compose): Transform for testing data.

    Returns:
        Tuple[Dataset, Dataset]: Train and test datasets.
    """
    train_size = int(CONFIG["train_split_ratio"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    return train_dataset, test_dataset
