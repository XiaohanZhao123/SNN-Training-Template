import random

import kornia
import kornia.image
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from kornia import augmentation
from torch import nn

from .auto_augument import POLICY, AutoArgumentation


def get_vision_dataset_augmentation(
    img_size: int, mean: list, std: list, autoaugment: bool = True
) -> tuple:
    """
    Get augmentation transforms for vision datasets.

    Args:
        img_size (int): The size to which images will be resized.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        autoaugment (bool, optional): Whether to use auto-augmentation. Defaults to True.

    Returns:
        tuple: A tuple containing two AugmentationSequential objects:
            - train_transform: Augmentation sequence for training data.
            - test_transform: Augmentation sequence for test data.
    """
    if autoaugment:
        auto_argument = _get_autoaugment_transform()
        train_transform = [
            augmentation.RandomResizedCrop(size=(img_size, img_size)),
            auto_argument,
        ]
    else:
        train_transform = [
            augmentation.RandomResizedCrop(size=(img_size, img_size)),
            augmentation.RandomHorizontalFlip(p=0.5),
            augmentation.RandomVerticalFlip(p=0.5),
            augmentation.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
            augmentation.RandomGrayscale(p=0.5),
            augmentation.RandomAffine(degrees=0, translate=(0, 0.2), p=0.7),
            augmentation.RandomAffine(degrees=0, scale=(0.2, 0), p=0.7),
            augmentation.RandomAffine(degrees=0, shear=(-30, 30), p=0.7),
            augmentation.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
            ),
        ]

    test_transform = [augmentation.Resize(size=(img_size, img_size))]
    normalize = [
        augmentation.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        Replication(T=4),
    ]

    train_transform = augmentation.ImageSequential(*train_transform, random_apply=1)
    train_transform = nn.Sequential(train_transform, *normalize)
    test_transform = augmentation.ImageSequential(*(test_transform + normalize))

    return train_transform, test_transform


def _get_autoaugment_transform():
    return AutoArgumentation(POLICY.CIFAR10)


class Replication(nn.Module):
    def __init__(self, T: int = 4) -> None:
        super().__init__()
        self.T = T

    def forward(self, x):
        return repeat(x, "b c h w -> t b c h w", t=self.T)


def get_neuromorphic_dataset_augmentation():
    """
    Get random augmentation transforms for neuromorphic datasets
    change the data format from BCTHW to BTCHW

    Returns:
        tuple: A tuple containing two nn.Sequential objects:
            - train_transform: Augmentation sequence for training data.
            - test_transform: Augmentation sequence for test data.
    """
    transform_list = [
        augmentation.RandomRotation(degrees=30),
        augmentation.RandomAffine(degrees=0, shear=(-30, 30)),
        nn.Identity(),
        augmentation.RandomHorizontalFlip(p=0.5),
        ImageRoll(max_shift=5),
        augmentation.RandomAffine(degrees=0, translate=(0, 0.2)),
        augmentation.RandomAffine(degrees=0, scale=(0.2, 0)),
    ]
    random_argumentation = augmentation.VideoSequential(
        *transform_list, same_on_frame=True, random_apply=1, data_format="BTCHW"
    )
    rearrange_layer = Rearrange("b t c h w -> t b c h w")
    train_transform = nn.Sequential(random_argumentation, rearrange_layer)
    test_transform = rearrange_layer
    return train_transform, test_transform


class ImageRoll(nn.Module):
    def __init__(self, max_shift) -> None:
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        shift_1 = random.randint(-self.max_shift, self.max_shift)
        shift_2 = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(x, shifts=(shift_1, shift_2), dims=(2, 3))
