from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..instantiate import CONFIG, get_dataset


class DataInterface(LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        try:
            dataset_config = self.config['dataset']
            self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
                dataset_name=dataset_config['name'],
                root=dataset_config['root'],
                img_size=dataset_config['img_size'],
                mean=dataset_config['mean'],
                std=dataset_config['std'],
                frame_number=dataset_config.get('frame_number', CONFIG['default_frames_number']),
                use_lmdb=dataset_config.get('use_lmdb', True),
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error setting up datasets: {str(e)}")

    def _get_dataloader(
        self, dataset: Optional[Dataset], batch_size: int, num_workers: int, shuffle: bool
    ):
        if dataset is None:
            return None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader(
            self.train_dataset,
            self.config["data_loader"]["batch_size"],
            self.config["data_loader"]["num_workers"],
            True,
        )

    def val_dataloader(self):
        return self._get_dataloader(
            self.val_dataset,
            self.config["data_loader"]["batch_size"],
            self.config["data_loader"]["num_workers"],
            False,
        )

    def test_dataloader(self):
        return self._get_dataloader(
            self.test_dataset,
            self.config["data_loader"]["batch_size"],
            self.config["data_loader"]["num_workers"],
            False,
        )
