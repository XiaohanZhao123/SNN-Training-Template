from typing import Mapping

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional
from torch import nn, optim
from torch.optim import lr_scheduler


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
        path: str,
        loss: Mapping,
        optimizer_kwargs: dict,
        scheduler_kwargs: dict,
        T: int,
        compile: bool = False,
    ) -> None:
        super().__init__()
        if compile is True:
            self.model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        else:
            self.model = model

        self.path = path
        self.loss_fn = loss
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.T = T

    def load_model(self):
        pass

    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        functional.reset_net(self)
        x = torch.stack([x] * self.T, dim=0)
        x = self.model(x)
        return x.mean(0)

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
        logits = self(x)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss, "train_acc": acc, "train_loss": loss}

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
        logits = self(img)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self) -> None:
        """
        Callback function called at the end of each validation epoch.
        """
        print("")

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
