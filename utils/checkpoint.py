"""
Utility functions for managing checkpoint loading into pure pytorch model

Key functions:
- get_ckpt_dir, get_ckpt_file: Locate checkpoint directories and files.
- check_config: Verify configuration settings.
- modify_checkpoint: Adapt checkpoints from different sources.
- load_ckpt: Load a checkpoint into a model.
- check_ckpt: Verify checkpoint structure.

Facilitates checkpoint handling across various training setups.
"""

from pathlib import Path

import torch
from omegaconf import DictConfig

__all__ = ["load_ckpt", "check_ckpt", "modify_checkpoint"]


def get_ckpt_dir(cfg: DictConfig):
    return f"./resources/models/{cfg.model.name}_{cfg.dataset.name}"


def get_ckpt_file(ckpt_dir: str):
    path = Path(ckpt_dir)
    # get the ckpt with the highest validation accuracy, ckpt with format best-epoch={epoch}-val_acc={val_acc}.ckpt
    ckpt_files = list(path.glob("best-epoch=*-val_acc=*.ckpt"))
    if len(ckpt_files) == 0:
        raise ValueError("No checkpoint files found in the directory")
    # sort the ckpt files by the validation accuracy
    ckpt_files.sort(key=lambda x: float(x.stem.split("-val_acc=")[1]))
    print(f"loading, {ckpt_files[-1]}")
    return ckpt_files[-1]


def check_config(cfg: DictConfig):
    if cfg.compile is True:
        assert cfg.to_pytorch is True, "to_pytorch must be True if compile is True"
        assert (
            cfg.debug is False
        ), "must set debug to False to fully utilize torch compile"


def _modify_checkpoint_from_compile(ckpt):
    """
    The checkpoint from torch compile is saved with the prefix model._orig_mod.
    This function modifies the checkpoint to remove the prefix and replace .model. with .
    """
    modified_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("model._orig_mod."):
            new_key = key.replace("model._orig_mod.", "model.")
            if ".model." in new_key:
                new_key = new_key.replace(".model.", ".")
            modified_ckpt[new_key] = value
        else:
            modified_ckpt[key] = value
    return modified_ckpt


def _modify_checkpoint_from_lightning(ckpt):
    """
    The checkpoint from lightning is saved with the prefix model.
    This function modifies the checkpoint to remove the prefix.
    """
    modified_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("model."):
            modified_ckpt[key.replace("model.", "")] = value
        elif not key.startswith("_train_transforms") and not key.startswith("_val_transforms"):
            modified_ckpt[key] = value
    return modified_ckpt


def modify_checkpoint(ckpt, from_compile=False, from_lightning=False):
    if from_compile:
        return _modify_checkpoint_from_compile(ckpt)
    elif from_lightning:
        return _modify_checkpoint_from_lightning(ckpt)
    else:
        return ckpt


def load_ckpt(model, cfg, from_compile=False, from_lightning=False):
    ckpt_dir = get_ckpt_dir(cfg)
    ckpt_file = get_ckpt_file(ckpt_dir)
    ckpt = torch.load(ckpt_file, map_location="cpu")["state_dict"]
    modified_ckpt = modify_checkpoint(
        ckpt, from_compile=from_compile, from_lightning=from_lightning
    )
    print(f"Loading checkpoint from {ckpt_file}")
    check_ckpt(modified_ckpt)
    model.load_state_dict(modified_ckpt)
    return model


def check_ckpt(ckpt):
    print("Checking modified checkpoint")
    for name, param in ckpt.items():
        print(f"{name}: {param.shape}")

    return ckpt
