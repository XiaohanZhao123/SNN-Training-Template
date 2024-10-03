import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import kornia.augmentation as K
import kornia.geometry as KG
import numpy as np
import torch
from kornia import augmentation as aug
from kornia.augmentation.auto import AutoAugment
from kornia.constants import Resample
from torch import nn
from torchvision.transforms import functional as TF

SUBPOLICY_CONFIG = List[Tuple[str, float, Optional[int]]]
IMAGENET_POLICY: List[SUBPOLICY_CONFIG] = [
    [("posterize", 0.4, 8), ("rotate", 0.6, 9)],
    [("solarize", 0.6, 5), ("auto_contrast", 0.6, None)],
    [("equalize", 0.8, None), ("equalize", 0.6, None)],
    [("posterize", 0.6, 7), ("posterize", 0.6, 6)],
    [("equalize", 0.4, None), ("solarize", 0.2, 4)],
    [("equalize", 0.4, None), ("rotate", 0.8, 8)],
    [("solarize", 0.6, 3), ("equalize", 0.6, None)],
    [("posterize", 0.8, 5), ("equalize", 1.0, None)],
    [("rotate", 0.2, 3), ("solarize", 0.6, 8)],
    [("equalize", 0.6, None), ("posterize", 0.4, 6)],
    [("rotate", 0.8, 8), ("color", 0.4, 0)],
    [("rotate", 0.4, 9), ("equalize", 0.6, None)],
    [("equalize", 0.0, None), ("equalize", 0.8, None)],
    [("invert", 0.6, None), ("equalize", 1.0, None)],
    [("color", 0.6, 4), ("contrast", 1.0, 8)],
    [("rotate", 0.8, 8), ("color", 1.0, 2)],
    [("color", 0.8, 8), ("solarize", 0.8, 7)],
    [("sharpness", 0.4, 7), ("invert", 0.6, None)],
    [("shear_x", 0.6, 5), ("equalize", 1.0, None)],
    [("color", 0.4, 0), ("equalize", 0.6, None)],
    [("equalize", 0.4, None), ("solarize", 0.2, 4)],
    [("solarize", 0.6, 5), ("auto_contrast", 0.6, None)],
    [("invert", 0.6, None), ("equalize", 1.0, None)],
    [("color", 0.6, 4), ("contrast", 1.0, 8)],
    [("equalize", 0.8, None), ("equalize", 0.6, None)],
]


CIFAR10_POLICY: List[SUBPOLICY_CONFIG] = [
    [("invert", 0.1, None), ("contrast", 0.2, 6)],
    [("rotate", 0.7, 2), ("translate_x", 0.3, 9)],
    [("sharpness", 0.8, 1), ("sharpness", 0.9, 3)],
    [("shear_y", 0.5, 8), ("translate_y", 0.7, 9)],
    [("auto_contrast", 0.5, None), ("equalize", 0.9, None)],
    [("shear_y", 0.2, 7), ("posterize", 0.3, 7)],
    [("color", 0.4, 3), ("brightness", 0.6, 7)],
    [("sharpness", 0.3, 9), ("brightness", 0.7, 9)],
    [("equalize", 0.6, None), ("equalize", 0.5, None)],
    [("contrast", 0.6, 7), ("sharpness", 0.6, 5)],
    [("color", 0.7, 7), ("translate_x", 0.5, 8)],
    [("equalize", 0.3, None), ("auto_contrast", 0.4, None)],
    [("translate_y", 0.4, 3), ("sharpness", 0.2, 6)],
    [("brightness", 0.9, 6), ("color", 0.2, 8)],
    [("solarize", 0.5, 2), ("invert", 0.0, None)],
    [("equalize", 0.2, None), ("auto_contrast", 0.6, None)],
    [("equalize", 0.2, None), ("equalize", 0.6, None)],
    [("color", 0.9, 9), ("equalize", 0.6, None)],
    [("auto_contrast", 0.8, None), ("solarize", 0.2, 8)],
    [("brightness", 0.1, 3), ("color", 0.7, 0)],
    [("solarize", 0.4, 5), ("auto_contrast", 0.9, None)],
    [("translate_y", 0.9, 9), ("translate_y", 0.7, 9)],
    [("auto_contrast", 0.9, None), ("solarize", 0.8, 3)],
    [("equalize", 0.8, None), ("invert", 0.1, None)],
    [("translate_y", 0.7, 9), ("auto_contrast", 0.9, None)],
]

SVHN_POLICY: List[SUBPOLICY_CONFIG] = [
    [("shear_x", 0.9, 4), ("invert", 0.2, None)],
    [("shear_y", 0.9, 8), ("invert", 0.7, None)],
    [("equalize", 0.6, None), ("solarize", 0.6, 6)],
    [("invert", 0.9, None), ("equalize", 0.6, None)],
    [("equalize", 0.6, None), ("rotate", 0.9, 3)],
    [("shear_x", 0.9, 4), ("auto_contrast", 0.8, None)],
    [("shear_y", 0.9, 8), ("invert", 0.4, None)],
    [("shear_y", 0.9, 5), ("solarize", 0.2, 6)],
    [("invert", 0.9, None), ("auto_contrast", 0.8, None)],
    [("equalize", 0.6, None), ("rotate", 0.9, 3)],
    [("shear_x", 0.9, 4), ("solarize", 0.3, 3)],
    [("shear_y", 0.8, 8), ("invert", 0.7, None)],
    [("equalize", 0.9, None), ("translate_y", 0.6, 6)],
    [("invert", 0.9, None), ("equalize", 0.6, None)],
    [("contrast", 0.3, 3), ("rotate", 0.8, 4)],
    [("invert", 0.8, None), ("translate_y", 0.0, 2)],
    [("shear_y", 0.7, 6), ("solarize", 0.4, 8)],
    [("invert", 0.6, None), ("rotate", 0.8, 4)],
    [("shear_y", 0.3, 7), ("translate_x", 0.9, 3)],
    [("shear_x", 0.1, 6), ("invert", 0.6, None)],
    [("solarize", 0.7, 2), ("translate_y", 0.6, 7)],
    [("shear_y", 0.8, 4), ("invert", 0.8, None)],
    [("shear_x", 0.7, 9), ("translate_y", 0.8, 3)],
    [("shear_y", 0.8, 5), ("auto_contrast", 0.7, None)],
    [("shear_x", 0.7, 2), ("invert", 0.1, None)],
]

POLICY_DICT = {
    "imagenet": IMAGENET_POLICY,
    "cifar10": CIFAR10_POLICY,
    "svhn": SVHN_POLICY,
}


class POLICY(enum.Enum):
    """
    Policy for AutoAugment.
    """
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


def get_trans(policy_dict: Dict[str, Any]) -> K.AugmentationBase2D:
    name = policy_dict["name"]
    p = policy_dict["p"]
    magnitude = policy_dict.get("magnitude")

    if name == "shear_x":
        return K.RandomAffine(degrees=0, shear=(magnitude, magnitude), p=p)
    elif name == "shear_y":
        return K.RandomAffine(degrees=0, shear=(0, 0, magnitude, magnitude), p=p)
    elif name == "translate_x":
        return K.RandomAffine(degrees=0, translate=(magnitude / 100, 0), p=p)
    elif name == "translate_y":
        return K.RandomAffine(degrees=0, translate=(0, magnitude / 100), p=p)
    elif name == "rotate":
        return K.RandomRotation(degrees=magnitude, p=p)
    elif name == "color":
        return K.ColorJitter(saturation=magnitude / 10, p=p)
    elif name == "posterize":
        return K.RandomPosterize(bits=8 - magnitude, p=p)
    elif name == "solarize":
        return K.RandomSolarize(thresholds=magnitude / 10, p=p)
    elif name == "contrast":
        return K.RandomContrast(magnitude / 10, p=p)
    elif name == "sharpness":
        return K.RandomSharpness(sharpness=magnitude / 10, p=p)
    elif name == "brightness":
        return K.RandomBrightness(brightness=magnitude / 10, p=p)
    elif name == "auto_contrast":
        return K.RandomAutoContrast(p=p)
    elif name == "equalize":
        return K.RandomEqualize(p=p)
    elif name == "invert":
        return K.RandomInvert(p=p)
    else:
        raise ValueError(f"Unknown operation: {name}")


class AutoArgumentation(torch.nn.Module):
    def __init__(self, policies: Union[List[SUBPOLICY_CONFIG], POLICY]):
        super().__init__()
        if isinstance(policies, POLICY):
            self.policies = POLICY_DICT[policies.value]
        else:
            self.policies = policies
        self.aug_policies = self._create_aug_policies()

    def _create_aug_policies(self):
        aug_policies = []
        for policy in self.policies:
            aug_list = []
            for op, p, magnitude in policy:
                aug = get_trans({"name": op, "p": p, "magnitude": magnitude})
                aug_list.append(aug)
            aug_policies.append(nn.Sequential(*aug_list))
        return K.AugmentationSequential(*aug_policies, random_apply=1)

    def forward(self, img):
        return self.aug_policies(img)


if __name__ == "__main__":
    sub_policy = AutoArgumentation(POLICY.CIFAR10)
    print(sub_policy)

    # check using random image
    for i in range(20):
        img = torch.randn(1, 3, 32, 32).abs().clamp(0, 1)
        img_aug = sub_policy(img)
        print(img_aug.shape)
