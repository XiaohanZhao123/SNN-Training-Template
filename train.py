import os
from copy import deepcopy
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks as plc
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model import ModuleInterface
from utils.data_interface import DataInterface
from utils.instantiate import get_augmentation, get_model

os.environ["WANDB_MODE"] = "disabled"
wandb.require("core")
torch.set_float32_matmul_precision('medium')


def load_callbacks(cfg: DictConfig):
    callbacks = []

    callbacks.append(
        plc.ModelCheckpoint(
            monitor="val_acc",
            filename="best-{epoch:02d}-{val_acc:.3f}",
            dirpath=f"./resources/models/{cfg.model.name}_{cfg.dataset.name}",
            save_weights_only=True,
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    )

    callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))

    return callbacks


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    seed_everything(cfg.seed)
    if cfg.use_wandb:
        logger = WandbLogger(
            config=config_dict,
            project="snn_model_design",
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
        )
    else:
        # create dir if not exists
        if not Path(f"./logs/{cfg.model.name}_{cfg.dataset.name}").exists():
            Path(f"./logs/{cfg.model.name}_{cfg.dataset.name}").mkdir(parents=True)
        logger = TensorBoardLogger(
            save_dir=f"./logs/{cfg.model.name}_{cfg.dataset.name}",
            name=cfg.wandb.run_name,
        )

    model_kwargs = deepcopy(config_dict["model"])
    del model_kwargs["name"]
    model = get_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
        to_pytorch=cfg.to_pytorch,
        T=cfg.T,
        **model_kwargs,
    )
    datamodule = DataInterface(config=config_dict)
    train_transforms, val_transforms = get_augmentation(
        dataset_name=cfg.dataset.name,
        img_size=cfg.dataset.img_size,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        autoaugment=cfg.autoaugment,
    )
    module = ModuleInterface(
        model=model,
        config=cfg,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )

    trainer = Trainer(
        logger=logger,
        callbacks=load_callbacks(cfg),
        max_epochs=cfg.max_epochs,
        devices=cfg.devices,
        precision=cfg.precision,
        num_sanity_val_steps=0 if not cfg.debug else 1,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
