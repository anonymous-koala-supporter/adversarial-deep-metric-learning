import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
from pkgs.module import DeepMetricLearning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")


def main(hparams):
    pl.seed_everything(hparams.seed)

    module = DeepMetricLearning(hparams)

    use_cuda = (not args.no_cuda) & torch.cuda.is_available()
    gpus = 1 if use_cuda else None

    log_path = f"{hparams.dataset}_{hparams.loss_kind}_{hparams.batch_size}"
    if hparams.adv_training:
        log_path += f"_{hparams.attack_rate}"

    logger = TensorBoardLogger(hparams.logdir, name=log_path,)

    callbacks = [
        ModelCheckpoint(
            monitor="precision_at_1", mode="max", filename="r_at_1-{epoch:02d}"
        )
    ]
    if hparams.adv_training:
        callbacks.append(
            ModelCheckpoint(
                monitor="precision_at_1_adv",
                mode="max",
                filename="r_at_1_adv-{epoch:02d}",
            )
        )

    trainer = pl.Trainer(
        default_root_dir=hparams.logdir,
        precision=hparams.precision,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=hparams.val_every_n_epoch,
        min_epochs=hparams.epochs,
        max_epochs=hparams.epochs,
        deterministic=True,
    )

    trainer.fit(module)
    trainer.test()


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(
        description="parser for Deep Metric Learning models"
    )
    main_arg_parser.add_argument(
        "--seed", type=int, default=0, help="random seed for training"
    )
    main_arg_parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="number of images after which the training loss is logged, default is 500",
    )
    main_arg_parser.add_argument(
        "--val-every-n-epoch", type=int, default=1,
    )
    main_arg_parser.add_argument(
        "--logdir",
        type=str,
        default="./metric_learning_logs",
        help="folder for tensorboard logs",
    )
    main_arg_parser.add_argument(
        "--precision", type=int, choices=[16, 32], default=32,
    )
    main_arg_parser.add_argument(
        "--no-cuda", action="store_true", default=False
    )
    main_arg_parser.add_argument(
        "--epochs", type=int, help="Number of epochs to run.", default=150
    )

    # add model specific args
    parser = DeepMetricLearning.add_model_specific_args(
        main_arg_parser, os.getcwd()
    )
    args = parser.parse_args()

    main(args)
