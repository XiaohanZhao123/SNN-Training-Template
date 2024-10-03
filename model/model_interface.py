from typing import Mapping, Optional, Union

import pytorch_lightning as pl
import torch
from kornia.augmentation import ImageSequential, VideoSequential
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics.classification.accuracy import Accuracy


class ModuleInterface(LightningModule):
    """
    A class representing the interface for a module in the SNN model design.

    Args:
        model (nn.Module): The neural network model.
        path (str): The path to save the model.
        loss (Mapping): The loss function.
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        scheduler_kwargs (dict): Keyword arguments for the scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Optional[Mapping] = None,
        config: Optional[DictConfig] = None,
        train_transforms: Union[ImageSequential, VideoSequential] = None,
        val_transforms: Union[ImageSequential, VideoSequential] = None,
    ) -> None:
        super().__init__()
        assert config is not None, "config is required"
        assert config.compile is not None, "config for compile is required"
        
        if config.compile is True:
            self.model = torch.compile(model, mode="reduce-overhead")
        else:
            self.model = model

        self.loss_fn = loss
        self.optimizer_kwargs = config.optimizer
        self.scheduler_kwargs = config.scheduler
        self._train_transforms = train_transforms
        self._val_transforms = val_transforms
        self.train_acc = Accuracy(
            task="multiclass", num_classes=config.dataset.num_classes, average="macro"
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=config.dataset.num_classes, average="macro"
        )
        
    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        functional.reset_net(self)
        output = self.model(x)
        return output

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        """
        Training step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the loss and training accuracy.
        """
        x, labels = batch
        x = self._train_transforms(x)
        logits = self(x)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, labels)

        metrics = {"loss": loss, "train_acc": self.train_acc}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the validation loss and accuracy.
        """
        img, labels = batch
        img = self._val_transforms(img)
        logits = self(img)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, labels)

        metrics = {"loss": loss, "val_acc": self.val_acc}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step of the module.

        Args:
            batch: The input batch.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the test loss and accuracy.
        """
        img, labels = batch
        img = self._val_transforms(img)
        logits = self(img)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, labels)

        metrics = {"loss": loss, "val_acc": self.val_acc}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Callback function called at the end of each validation epoch.
        """
        print("epoch end")

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self.optimizer_kwargs["name"])
        assert (
            optimizer_cls is not None
        ), f"Optimizer {self.optimizer_kwargs['name']} not found"

        optimizer = optimizer_cls(
            self.model.parameters(), **self.optimizer_kwargs["params"]
        )

        scheduler_cls = getattr(lr_scheduler, self.scheduler_kwargs["name"])
        assert (
            scheduler_cls is not None
        ), f"Scheduler {self.scheduler_kwargs['name']} not found"
        scheduler = scheduler_cls(optimizer, **self.scheduler_kwargs["params"])
        scheduler_dict = {
            "scheduler": scheduler,
            "name": "lr-scheduler",
            "interval": "epoch",
        }

        return [optimizer], [scheduler_dict]
