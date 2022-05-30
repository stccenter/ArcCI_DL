import os
import shutil
from typing import Tuple, Dict

import albumentations as A
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from IceClassifier.preprocessing.data_module import common_transforms
from IceClassifier.preprocessing.dataset import IceTilesDataset
from IceClassifier.preprocessing.utils import make_dir

NUM_WORKERS = min(os.cpu_count(), 4)


class LogExamplesCallback(Callback):
    def __init__(
            self,
            dataset_name: str,
            class_labels: Dict[int, str],
            tmp_dir: str = './tmp_ice_data',
            tile_shape: Tuple[int, int] = (256, 256),
            batch_size: int = 16,
    ):
        self.dataset_name = dataset_name
        self.class_labels = class_labels
        self.tmp_dir = tmp_dir
        self.tile_shape = tile_shape
        self.batch_size = batch_size

    def on_init_start(self, trainer):
        make_dir(self.tmp_dir)
        output_dir = self.__fetch_examples()
        self.__build_dataloader(output_dir)

    def __fetch_examples(self):
        dataset_dir_root = f'{self.tmp_dir}/{self.dataset_name}'
        make_dir(dataset_dir_root)
        artifact = wandb.use_artifact(f'{self.dataset_name}:latest')
        artifact_dir = artifact.download(root=dataset_dir_root)
        return f'{artifact_dir}/{self.tile_shape[0]}x{self.tile_shape[1]}'

    def __build_dataloader(self, output_dir):
        examples_transforms = A.Compose([*common_transforms])
        self.examples_dataset = IceTilesDataset(output_dir, examples_transforms)
        self.examples_dataloader = DataLoader(
            self.examples_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            shuffle=False
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        examples = []
        for batch in self.examples_dataloader:
            batch = {k: v.to(device=pl_module.device) for k, v in batch.items()}
            prediction = pl_module(batch)
            prediction_mask = torch.argmax(prediction, dim=1)
            for i in range(len(batch)):
                image = batch['image'][i]
                mask = batch['mask'][i]
                pred = prediction_mask[i]

                mask = mask.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()

                masks = {
                    "prediction": {"mask_data": pred, "class_labels": self.class_labels},
                    "ground_truth": {"mask_data": mask, "class_labels": self.class_labels}
                }
                artifact = wandb.Image(image, masks=masks)
                examples.append(artifact)

        wandb.log({
            "val/examples": examples,
            "global_step": trainer.global_step
        })

    def teardown(self, trainer, pl_module, stage=None):
        shutil.rmtree(f'{self.tmp_dir}/{self.dataset_name}', ignore_errors=True)
