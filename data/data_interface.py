from typing import Type

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from copy import deepcopy


class DataInterface(LightningDataModule):
    """
    DataInterface class for managing datasets and dataloaders in a Lightning module.

    Args:
        dataset_cls (Type[Dataset]): The dataset class to be used.
        dataset_kwargs (dict): Keyword arguments to be passed to the dataset class.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for data loading.
        has_test (bool): Whether the dataset has a test set.

    Attributes:
        dataset_cls (Type[Dataset]): The dataset class to be used.
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): The number of workers for data loading.
        has_test (bool): Whether the dataset has a test set.
        dataset_kwargs (dict): Keyword arguments to be passed to the dataset class.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.

    Methods:
        setup(stage: str) -> None: Setup method to initialize the datasets.
        train_dataloader() -> DataLoader: Returns the training dataloader.
        val_dataloader() -> DataLoader: Returns the validation dataloader.
        test_dataloader() -> DataLoader: Returns the test dataloader.
        prepare_data() -> None: Method to prepare the data.

    """

    def __init__(
        self,
        dataset_cls: Type[Dataset],
        dataset_kwargs: dict,
        batch_size: int,
        num_workers: int,
        has_test: bool,
    ) -> None:
        super().__init__()
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.has_test = has_test
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage: str) -> None:
        self.train_dataset, self.val_dataset, self.test_dataset = _get_datasets(
            self.dataset_cls, self.dataset_kwargs, self.has_test
        )
        if self.test_dataset is None:
            _, self.test_dataset = random_split(
                self.train_dataset,
                [
                    int(0.9 * len(self.val_dataset)),
                    len(self.val_dataset) - int(0.9 * len(self.val_dataset)),
                ],
            )
            if hasattr(self.train_dataset.dataset, "transform"):
                self.test_dataset.dataset.transform = self.dataset_kwargs["test"][
                    "transform"
                ]

    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def prepare_data(self) -> None:
        _ = self.dataset_cls(
            root=self.dataset_kwargs["train"]["root"], train=True, download=True
        )

        if self.has_test:
            _ = self.dataset_cls(
                root=self.dataset_kwargs["train"]["root"], train=False, download=True
            )


def _get_datasets(dataset_cls: Type[Dataset], dataset_kwargs: dict, has_test: bool):

    train_dataset = dataset_cls(train=True, **dataset_kwargs["train"])
    # test_dataset = dataset_cls(train=False, **dataset_kwargs["test"])
    # val_dataset = test_dataset
    # just try the paper's method

    train_dataset, val_dataset = random_split(
        train_dataset,
        [
            int(0.9 * len(train_dataset)),
            len(train_dataset) - int(0.9 * len(train_dataset)),
        ],
    )
    if hasattr(train_dataset.dataset, "transform"):
        # prevent modification on original dataset
        val_dataset = deepcopy(val_dataset)
        val_dataset.dataset.transform = dataset_kwargs["test"]["transform"]

    if has_test:
        test_dataset = dataset_cls(train=False, **dataset_kwargs["test"])
        return train_dataset, val_dataset, test_dataset

    return train_dataset, test_dataset, test_dataset
