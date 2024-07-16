import os

os.environ["WANDB_MODE"] = "disabled"
import wandb

wandb.require("core")
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks as plc
from pytorch_lightning.loggers import WandbLogger

from data import DataInterface
from model import ModuleInterface
from utils import get_dataset, get_loss, get_model


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
    logger = WandbLogger(
        config=config_dict,
        project="snn_model_design",
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
    )
    model = get_model(cfg)
    dataset_cls, dataset_kwargs = get_dataset(cfg)
    loss = get_loss(cfg)
    model = ModuleInterface(
        model,
        path=cfg.model.path,
        loss=loss,
        optimizer_kwargs=cfg.optimizer,
        scheduler_kwargs=cfg.scheduler,
        T=cfg.T,
        compile=cfg.compile,
    )
    dataset = DataInterface(
        dataset_cls=dataset_cls,
        dataset_kwargs=dataset_kwargs,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        has_test=cfg.data.has_test,
    )

    print(model)
    print(dataset_cls, dataset_kwargs)

    trainer = Trainer(
        logger=logger,
        callbacks=load_callbacks(cfg),
        **cfg.trainer,
    )
    # model = torch.compile(model)
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()


"""cli

nohup python train.py data.auto_argu=True data.cutout=False replace_bn=False clamp_activation=False trainer.devices=[1] wandb.run_name=with_auto_argu >with_auto_argu.out &

nohup python train.py data.auto_argu=False data.cutout=False replace_bn=False clamp_activation=False trainer.devices=[0] wandb.run_name=baseline >baseline.out &

nohup python train.py data.auto_argu=True data.cutout=True replace_bn=False clamp_activation=False trainer.devices=[2] wandb.run_name=with_full_argu >with_full_argu.out &

nohup python train.py data.auto_argu=True data.cutout=True replace_bn=True clamp_activation=True trainer.devices=[3] wandb.run_name=full >full.out &
"""
