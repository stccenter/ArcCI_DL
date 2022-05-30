import os
import shutil
from typing import Optional, Tuple

import albumentations as A
import pytorch_lightning as pl
import wandb
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from IceClassifier.preprocessing.dataset import IceTilesDataset
from IceClassifier.preprocessing.utils import make_dir

common_transforms = [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]

NUM_WORKERS = min(os.cpu_count(), 4)


class IceTilesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tmp_dir: str = './tmp_ice_data',
            dataset_name: Optional[str] = None,
            tile_shape: Optional[Tuple[int, int]] = (256, 256),
            batch_size: int = 16
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tmp_dir = tmp_dir
        self.tile_shape = tile_shape
        self.batch_size = batch_size
        self.cloud_dataset = dataset_name is not None
        self.output_dir = tmp_dir if not self.cloud_dataset else None

    def setup(self, stage: Optional[str] = None):
        if self.cloud_dataset:
            make_dir(self.tmp_dir)
            self.__fetch_tiles()

    def __fetch_tiles(self):
        dataset_dir_root = f'{self.tmp_dir}/{self.dataset_name}'
        make_dir(dataset_dir_root)
        artifact = wandb.use_artifact(f'{self.dataset_name}:latest')
        artifact_dir = artifact.download(root=dataset_dir_root)
        self.output_dir = f'{artifact_dir}/{self.tile_shape[0]}x{self.tile_shape[1]}'

    def __create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, pin_memory=True)

    def train_dataloader(self):
        train_transforms = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                *common_transforms
            ]
        )
        train_dataset = IceTilesDataset(f'{self.output_dir}/train', train_transforms)
        return self.__create_dataloader(train_dataset)

    def val_dataloader(self):
        val_transforms = A.Compose([
            *common_transforms
        ])
        val_dataset = IceTilesDataset(f'{self.output_dir}/val', val_transforms)
        return self.__create_dataloader(val_dataset)

    def test_dataloader(self):
        test_transforms = A.Compose([
            *common_transforms
        ])
        test_dataset = IceTilesDataset(f'{self.output_dir}/test', test_transforms)
        return self.__create_dataloader(test_dataset)

    def teardown(self, stage=None):
        if self.cloud_dataset:
            shutil.rmtree(f'{self.tmp_dir}/{self.dataset_name}', ignore_errors=True)
            shutil.rmtree(self.output_dir, ignore_errors=True)
