import os
import shutil
from typing import Optional, Tuple

import uuid

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader

from IceClassifier.preprocessing.dataset import IceTilesDataset
from IceClassifier.preprocessing.image_splitter import ImageSplitter
from IceClassifier.preprocessing.utils import make_dir

common_transforms = [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]

NUM_WORKERS = 4


class IceTilesDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = '/att/nobackup/kswang/newSemanticSegmentation/resources/datasets/original',
            tmp_dir: str = '/att/nobackup/kswang/newSemanticSegmentation/tmp_ice_data',
            tile_shape: Tuple[int, int] = (256, 256),
            batch_size: int = 16
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir + '/' + uuid.uuid4().hex.upper()[0:6]
        self.tile_shape = tile_shape
        self.output_dir = None
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.output_dir = f'{self.tmp_dir}/{self.tile_shape[0]}x{self.tile_shape[1]}'
        make_dir(self.tmp_dir)
        make_dir(self.output_dir)
        splitter = ImageSplitter(self.data_dir)
        splitter.transform(self.tile_shape, self.output_dir)

    def __create_dataloader(self, dataset, num_workers=NUM_WORKERS):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers, pin_memory=True)

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
        return self.__create_dataloader(val_dataset, 1)

    def test_dataloader(self):
        test_transforms = A.Compose([
            *common_transforms
        ])
        test_dataset = IceTilesDataset(f'{self.output_dir}/test', test_transforms)
        return self.__create_dataloader(test_dataset, 1)

    def teardown(self, stage: Optional[str] = None):
        if self.output_dir is not None:
            shutil.rmtree(self.output_dir)
