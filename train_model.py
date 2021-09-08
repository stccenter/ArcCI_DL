import uuid
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

import wandb
from IceClassifier.preprocessing.data_module import IceTilesDataModule
from IceClassifier.preprocessing.utils import make_dir
from IceClassifier.training.callbacks.log_examples import LogExamplesCallback
from IceClassifier.training.system import IceClassifierSystem

EXPERIMENT_SEED = 0
TILE_SHAPE = (256, 256)

SEG_CLASSES = {
    0: "Background",
    1: "Thick Ice",
    2: "Thin Ice",
    3: "Shadow",
    4: "Open Water",
    5: "Unknown"
}
NUM_CLASSES = len(SEG_CLASSES)
LOSS_WEIGHTS = {
    "original": [1., 1.13, 6.1, 27.36, 21.50, 100.],
    "badlight": [1., 1.03, 23.02, 1186.48, 62.58, 100.]
}

TMP_DIR = './resources/tmp_ice_data'
TENSORBOARD_DIR = './resources/logs'
CHECKPOINT_DIR = './resources/models'

WANDB_PROJECT = 'IceClassification'
WANDB_ENTITY = 'semanticsegmentation'


def main():
    pl.seed_everything(EXPERIMENT_SEED)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = IceClassifierSystem.add_model_specific_args(parser)
    args = parser.parse_args()

    is_local_mode = args.local_mode
    if not is_local_mode:
        wandb.init(
            job_type='model-training',
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            settings=wandb.Settings(start_method="thread")
        )

    run_id = generate_run_id()
    create_folders(run_id)
    system, tiles_dm = create_system_and_dataset(run_id, args)

    callbacks = create_callbacks(run_id, args, save_checkpoints=is_local_mode, save_examples=not is_local_mode)
    loggers = create_loggers(run_id, system, use_tensorboard=is_local_mode, use_wandb=not is_local_mode)

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=loggers, deterministic=True)
    trainer.fit(system, tiles_dm)


def generate_run_id():
    long_id = uuid.uuid4().hex
    return long_id[:8]


def create_folders(run_id):
    make_dir('./resources')
    make_dir(TMP_DIR)
    make_dir(f'{TMP_DIR}/{run_id}')


def create_system_and_dataset(run_id, args):
    loss_weights = LOSS_WEIGHTS.get(args.dataset, [1. for _ in range(NUM_CLASSES)])
    tiles_dm = __create_dataset(run_id, args)
    system = IceClassifierSystem(
        num_classes=NUM_CLASSES,
        backbone_model=args.backbone_model,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        loss_weights=loss_weights
    )
    return system, tiles_dm


def __create_dataset(run_id, args):
    if args.local_mode:
        return IceTilesDataModule(
            tmp_dir=args.dataset_path,
            dataset_name=None,
            tile_shape=None,
            batch_size=args.batch_size
        )
    else:
        dataset_name = f'ice-tiles-dataset-{args.dataset}'
        return IceTilesDataModule(
            tmp_dir=f'{TMP_DIR}/{run_id}',
            dataset_name=dataset_name,
            tile_shape=TILE_SHAPE,
            batch_size=args.batch_size
        )


def create_callbacks(run_id, args, save_checkpoints=True, save_examples=True):
    callbacks = []
    if save_checkpoints:
        models_dir = f'{CHECKPOINT_DIR}/{run_id}'
        checkpoints_callback = pl_callbacks.ModelCheckpoint(dirpath=models_dir)
        callbacks.append(checkpoints_callback)
    if save_examples:
        dataset_name = f'ice-tiles-examples-{args.dataset}'
        log_images_callback = LogExamplesCallback(
            tmp_dir=f'{TMP_DIR}/{run_id}',
            dataset_name=dataset_name,
            class_labels=SEG_CLASSES,
            tile_shape=TILE_SHAPE,
            batch_size=args.batch_size
        )
        callbacks.append(log_images_callback)
    return callbacks


def create_loggers(run_id, system, use_tensorboard=True, use_wandb=True):
    loggers = []
    if use_tensorboard:
        tensorboard_dir = f'{TENSORBOARD_DIR}/{run_id}'
        tb_logger = pl_loggers.TensorBoardLogger(tensorboard_dir)
        loggers.append(tb_logger)
    if use_wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            log_model='all',
            settings=wandb.Settings(start_method="thread")
        )
        wandb_logger.watch(system)
        loggers.append(wandb_logger)
    return loggers


if __name__ == '__main__':
    main()
