import os
import uuid
import wandb
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

from IceClassifier.preprocessing.data_module import IceTilesDataModule
from IceClassifier.training.system import IceClassifierSystem

NUM_CLASSES = 6
MODEL_PREFIX = uuid.uuid4().hex.upper()[0:6]
TENSORBOARD_DIR = './resources/logs' + '/' + MODEL_PREFIX
CHECKPOINT_DIR = './resources/models' + '/' + MODEL_PREFIX

WANDB_PROJECT = 'IceClassification'
WANDB_ENTITY = 'semanticsegmentation'


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = IceClassifierSystem.add_model_specific_args(parser)
    args = parser.parse_args()

    system, tiles_dm = create_system_and_dataset(args)
    callbacks = create_callbacks()
    loggers = create_loggers()
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=loggers)

    trainer.fit(system, tiles_dm)
    loggers[1].watch(system)


def create_system_and_dataset(args):
    tiles_dm = IceTilesDataModule(batch_size=args.batch_size)
    system = IceClassifierSystem(
        num_classes=NUM_CLASSES,
        backbone_model=args.backbone_model,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate
    )
    return system, tiles_dm


def create_callbacks():
    checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath=CHECKPOINT_DIR)
    return [checkpoint_callback]


def create_loggers():
    tb_logger = pl_loggers.TensorBoardLogger(TENSORBOARD_DIR)
    wandb_logger = pl_loggers.WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model='all',
	settings=wandb.Settings(start_method='fork')
    )
    return [tb_logger, wandb_logger]


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    main()
