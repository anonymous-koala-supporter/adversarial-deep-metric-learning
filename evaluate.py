import argparse
import json
import os
import warnings
from glob import glob

import pytorch_lightning as pl
from pkgs.module import DeepMetricLearning
from pkgs.utils import checkpoint_by_name
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings("ignore")


def main(hparams):
    pl.seed_everything(hparams.seed)

    if hparams.model_dir != "":
        checkpoints = glob(
            os.path.join(hparams.model_dir, "checkpoints", "*.ckpt")
        )

        ckpt_path = checkpoint_by_name(checkpoints, hparams.ckpt)
        module = DeepMetricLearning.load_from_checkpoint(
            checkpoint_path=ckpt_path, **vars(hparams), only_eval=True
        )

    else:
        module = DeepMetricLearning(hparams, only_eval=True)

    use_cuda = not args.no_cuda
    gpus = 1 if use_cuda else None

    trainer = pl.Trainer(
        gpus=gpus, precision=hparams.precision, deterministic=True
    )
    test_output = trainer.test(module, verbose=False)[0]
    print(json.dumps(test_output, indent=2))


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(
        description="parser for FaceNet models"
    )
    main_arg_parser.add_argument(
        "--model-dir", type=str, default="", help="directory for model"
    )
    main_arg_parser.add_argument(
        "--ckpt", type=str, default="", help="specify checkpoint to use"
    )
    main_arg_parser.add_argument(
        "--seed", type=int, default=0, help="random seed for training"
    )
    main_arg_parser.add_argument(
        "--precision", type=int, choices=[16, 32], default=32,
    )
    main_arg_parser.add_argument(
        "--no-cuda", action="store_true", default=False
    )
    # add model specific args
    parser = DeepMetricLearning.add_model_specific_args(
        main_arg_parser, os.getcwd()
    )
    args = parser.parse_args()

    main(args)
