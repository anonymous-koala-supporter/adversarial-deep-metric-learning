import argparse
import os

import numpy as np

import pytorch_lightning as pl
import torch
from pkgs.attacks import (
    CWAttack,
    FGSMAttack,
    GlobalAttack,
    PGDAttack,
    freeze_model,
)
from pkgs.attacks.norms import L2NormBall, LinfNormBall
from pkgs.datasets import (
    CARS196Dataset,
    ClassBalancedSampler,
    CUB200Dataset,
    GaussianDataset,
    SOPDataset,
    VisualPhishDataset,
)
from pkgs.losses import ReferenceDistanceLoss
from pkgs.models import ResNet50, VisualPhishNet
from pkgs.utils import SearchIndex, get_ith_or_first, max_per_anchor
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
)
from torchvision import transforms


class DeepMetricLearning(pl.LightningModule):
    def __init__(self, hparams, only_eval=False):
        super(DeepMetricLearning, self).__init__()
        self.only_eval = only_eval

        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.hparams = hparams

        if hparams.model == "resnet":
            pretrained = hparams.pretrained != "none"
            self.model = ResNet50(
                hparams.embedding_size,
                pretrained=pretrained,
                normalize_last=not hparams.no_normalization,
            )
        elif hparams.model == "visualphishnet":
            pretrained = (
                False if hparams.pretrained is "none" else hparams.pretrained
            )
            self.model = VisualPhishNet(
                hparams.embedding_size,
                pretrained=pretrained,
                normalize_last=not hparams.no_normalization,
            )
        else:
            raise ValueError(f"unknown model: {hparams.model}")

        self.__init_loss_fn(hparams)
        self.__init_attack(hparams)

    def __init_loss_fn(self, hparams):
        self.miner = miners.DistanceWeightedMiner(
            cutoff=0.5, nonzero_loss_cutoff=1.4
        )
        if hparams.loss_kind == "contrastive":
            self.loss_fn = losses.ContrastiveLoss(
                neg_margin=hparams.contrastive_neg_margin
            )
        elif hparams.loss_kind == "triplet":
            self.loss_fn = losses.TripletMarginLoss(margin=hparams.margin)
        else:
            raise ValueError(f"unknown loss: {hparams.loss_kind}")

    def __init_attack(self, hparams):
        self.adv_training = False

        attacks = {}
        global_attacks = {}
        if hparams.attack != "none":
            n_epsilons = len(hparams.epsilon.split(","))
            for i, epsilon in enumerate(hparams.epsilon.split(",")):
                epsilon_num = float(epsilon)

                additional_params = {}
                if hparams.attack == "fgsm":
                    params = {"fgsm_alpha": ("alpha", float)}
                    attack_init = FGSMAttack
                if hparams.attack == "r-fgsm":
                    params = {"fgsm_alpha": ("alpha", float)}
                    additional_params["random_init"] = True
                    attack_init = FGSMAttack
                elif hparams.attack == "pgd":
                    params = {
                        "pgd_steps": ("iterations", int),
                        "pgd_random_restarts": ("random_restarts", int),
                        "pgd_alpha": ("alpha", float),
                    }
                    attack_init = PGDAttack
                elif hparams.attack == "cw":
                    attack_init = CWAttack
                    params = {
                        "cw_const": ("init_const", float),
                        "cw_lr": ("learning_rate", float),
                        "cw_df": ("decrease_factor", float),
                        "cw_iterations": ("iterations", int),
                    }

                attack_kwargs = {
                    k: get_ith_or_first(
                        getattr(hparams, paramkey), i, n_epsilons, type=vtype
                    )
                    for (paramkey, (k, vtype)) in params.items()
                }

                for k, v in additional_params.items():
                    attack_kwargs[k] = v

                if hparams.norm == "linf":
                    norm = LinfNormBall(epsilon_num)
                elif hparams.norm == "l2":
                    norm = L2NormBall(epsilon_num)

                attacks[epsilon] = attack_init(norm, **attack_kwargs)

                if self.only_eval:
                    # use attack specified by params for evaluation
                    global_attacks[epsilon] = GlobalAttack(
                        attack_init(norm, **attack_kwargs)
                    )
                else:
                    # default to PGD if not evaluation
                    global_attacks[epsilon] = GlobalAttack(
                        PGDAttack(norm, iterations=5)
                    )

        self.attacks = attacks if len(attacks) > 0 else None
        self.global_attacks = (
            global_attacks if len(global_attacks) > 0 else None
        )

        if hparams.adv_training:
            if len(self.attacks) != 1:
                raise ValueError(
                    f"Can't perform adversarial training with multiple attacks"
                )

            self.attack = self.attacks[list(self.attacks.keys())[0]]
            self.adv_training = True
            self.attack_rate = hparams.attack_rate

    def _get_dataset(self, train=True):
        ds_fn = {
            "cub200": CUB200Dataset,
            "cars196": CARS196Dataset,
            "sop": SOPDataset,
            "synthetic": GaussianDataset,
            "visualphish": VisualPhishDataset,
        }[self.hparams.dataset]

        dataset_folder = os.path.join(
            self.hparams.data_dir, self.hparams.dataset
        )

        transform = None
        if self.hparams.dataset == "visualphish":
            transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(),]
            )

        return ds_fn(
            dataset_folder,
            train=train,
            split=self.hparams.data_split,
            split_kind=self.hparams.data_split_kind,
            transform=transform,
            return_idx=True,
        )

    def forward(self, x):
        output = self.model.forward(x)
        return output

    def training_step(self, batch, _):
        X, labels, _ = batch
        embeddings = self.forward(X)

        with torch.no_grad():
            miner_output = self.miner(embeddings, labels)
            miner_output = max_per_anchor(
                miner_output, self.hparams.n_per_anchor
            )

        if self.adv_training:
            with torch.no_grad():
                loss_benign = self.loss_fn(embeddings, labels, miner_output)
                self.log("loss", loss_benign)

            n = len(miner_output[0])
            idxs_to_attack = (torch.rand(n) < self.hparams.attack_rate).bool()
            if idxs_to_attack.any():
                ref_loss_kwargs = {}
                per_miner_idx = None
                if self.hparams.perturb_target == "pos":
                    pos_idxs = miner_output[0][idxs_to_attack]
                    ref_loss_kwargs["pos_embs"] = embeddings[pos_idxs]
                    per_miner_idx = 1
                elif self.hparams.perturb_target == "neg":
                    neg_idxs = miner_output[0][idxs_to_attack]
                    ref_loss_kwargs["neg_embs"] = embeddings[neg_idxs]
                    per_miner_idx = 2
                elif self.hparams.perturb_target == "anc":
                    pos_idxs = miner_output[1][idxs_to_attack]
                    ref_loss_kwargs["pos_embs"] = embeddings[pos_idxs]
                    neg_idxs = miner_output[2][idxs_to_attack]
                    ref_loss_kwargs["neg_embs"] = embeddings[neg_idxs]
                    per_miner_idx = 0

                attack_loss_fn = ReferenceDistanceLoss(**ref_loss_kwargs)

                samples_to_attack = miner_output[per_miner_idx][idxs_to_attack]
                X_adv = X[samples_to_attack].detach().clone()
                labels_adv = labels[samples_to_attack]

                unfreeze = freeze_model(self.model)
                X_adv = self.attack(
                    X_adv, labels_adv, self.model, attack_loss_fn
                )
                unfreeze()

                miner_output = list(miner_output)
                miner_output[per_miner_idx][idxs_to_attack] = torch.arange(
                    len(X_adv)
                ).to(self.device) + len(
                    X
                )  # append newly perturbed data points

                miner_output = tuple(miner_output)

                with torch.no_grad():
                    X = torch.cat([X, X_adv])
                    labels = torch.cat([labels, labels_adv])

            old_loss = loss_benign
            loss = self.loss_fn(self.model(X), labels, miner_output)
            self.log("loss_adv", loss)
        else:
            loss = self.loss_fn(embeddings, labels, miner_output)
            self.log("loss", loss)

        return loss

    def on_validation_epoch_start(self):
        if self.attacks is None and not self.hparams.infer_from_train:
            return

        if not hasattr(self, "data_index"):
            self.data_index = SearchIndex(self.hparams.embedding_size)
        else:
            self.data_index.reset()

        populate_index = True
        if self.hparams.use_cache_index:
            found = self.data_index.load(self.hparams)
            populate_index = found is False

        if populate_index:
            ref_dataset = self._get_dataset(
                train=self.hparams.infer_from_train
            )
            loader = torch.utils.data.DataLoader(
                ref_dataset,
                batch_size=self.hparams.batch_size * 4,
                num_workers=self.hparams.num_workers,
            )

            for (X, labels, _) in iter(loader):
                embeddings = self.forward(X.to(self.device))
                self.data_index.add(embeddings.cpu().numpy(), labels)

        if self.hparams.use_cache_index and populate_index:
            self.data_index.save(self.hparams)

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def validation_step(self, batch, _):
        X, labels, idxs_X = batch
        embeddings = self.forward(X)

        metrics = {
            "embeddings": embeddings,
            "labels": labels,
        }

        if self.attacks is None:
            return metrics

        unfreeze = freeze_model(self.model)

        for eps, gattack in self.global_attacks.items():
            X_adv = gattack(self.data_index, X, idxs_X, labels, self.model)

            embeddings_adv = self.forward(X_adv)

            suffix = f"eps={eps}"
            metrics[f"embeddings_{suffix}"] = embeddings_adv.cpu()

            del embeddings_adv

        unfreeze()

        return metrics

    def validation_epoch_end(self, outputs):
        embeddings = (
            torch.cat([x["embeddings"] for x in outputs]).cpu().numpy()
        )
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        acc_calc = AccuracyCalculator()

        ref_embeddings = (
            embeddings
            if not self.hparams.infer_from_train
            else self.data_index.retrieve(range(len(self.data_index)))
        )
        ref_labels = (
            labels
            if not self.hparams.infer_from_train
            else self.data_index.labels.cpu().numpy()
        )
        same_source = not self.hparams.infer_from_train

        torch.cuda.empty_cache()
        metrics = acc_calc.get_accuracy(
            embeddings, ref_embeddings, labels, ref_labels, same_source
        )

        metrics["n_samples"] = len(labels)

        if self.attacks is not None:
            for epsilon, _ in self.attacks.items():
                tmp_embeddings = (
                    torch.cat(
                        [x[f"embeddings_eps={epsilon}"] for x in outputs]
                    )
                    .cpu()
                    .numpy()
                )

                tmp_metrics = acc_calc.get_accuracy(
                    tmp_embeddings,
                    ref_embeddings,
                    labels,
                    ref_labels,
                    same_source,
                )

                if self.adv_training:
                    for k, v in tmp_metrics.items():
                        metrics[f"{k}_adv"] = v
                else:
                    for k, v in tmp_metrics.items():
                        metrics[f"{k}_eps={epsilon}"] = v

        for k, v in metrics.items():
            self.log(k, v)

        if hasattr(self, "data_index"):
            self.data_index.reset()

    def configure_optimizers(self):
        opts = [
            {
                "params": self.model.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            }
        ]

        loss_optim_opts = getattr(self, "loss_optim_opts", None)
        if loss_optim_opts is not None:
            opts += [loss_optim_opts]

        return torch.optim.Adam(opts)

    def train_dataloader(self):
        train_dataset = self._get_dataset(train=True)

        labels = train_dataset.get_labels()
        sampler = ClassBalancedSampler(
            labels,
            self.hparams.batch_size,
            self.hparams.samples_per_class,
            strict=not self.hparams.no_strict_sampling,
        )

        kwargs = (
            {"num_workers": self.hparams.num_workers, "pin_memory": True}
            if not self.hparams.no_cuda
            else {}
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=sampler, **kwargs,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self._get_dataset(train=False)

        kwargs = (
            {"num_workers": self.hparams.num_workers, "pin_memory": False}
            if not self.hparams.no_cuda
            else {}
        )

        labels = val_dataset.get_labels()
        sampler = ClassBalancedSampler(
            labels,
            self.hparams.batch_size,
            self.hparams.samples_per_class,
            strict=False,
            resample=False,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_sampler=sampler, **kwargs
        )

        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    @staticmethod
    def add_model_specific_args(parser, _root_dir):
        parser.add_argument(
            "--model",
            type=str.lower,
            default="resnet",
            choices=["resnet", "visualphishnet"],
        )
        parser.add_argument(
            "--pretrained",
            type=str.lower,
            default="",
            choices=["", "none", "natural", "robust"],
        )
        parser.add_argument(
            "--no-normalization", action="store_true", default=False
        )
        parser.add_argument(
            "--dataset",
            type=str.lower,
            default="cub200",
            choices=["cub200", "cars196", "sop", "synthetic", "visualphish"],
        )
        parser.add_argument(
            "--data-dir", type=str, default="input",
        )
        parser.add_argument("--data-split", default="sorted")
        parser.add_argument(
            "--data-split-kind",
            default="isolated",
            choices=["isolated", "random"],
        )
        parser.add_argument(
            "--batch-size", type=int, default=112,  # from Roth et al.
        )
        parser.add_argument(
            "--samples-per-class", type=int, default=2,
        )
        parser.add_argument(
            "--margin", type=float, default=0.2,
        )
        parser.add_argument(
            "--embedding-size", type=int, default=128,
        )
        parser.add_argument(
            "--loss-kind",
            type=str,
            choices=["triplet", "contrastive"],
            default="contrastive",
        )

        parser.add_argument(
            "--no-strict-sampling", action="store_true", default=False
        )
        parser.add_argument(
            "--infer-from-train", action="store_true", default=False
        )
        parser.add_argument(
            "--n-per-anchor", type=int, default=1,
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            help="Number of workers for dataloader",
            default=4,
        )
        parser.add_argument(
            "--lr", type=float, help="Learning rate", default=1e-5
        )
        parser.add_argument("--weight_decay", type=float, default=4e-4)

        # losses
        parser.add_argument(
            "--contrastive-neg-margin",
            type=float,
            help="Margin for negative pairs of contrastive loss",
            default=1,
        )

        # attacks
        parser.add_argument(
            "--attack",
            type=str,
            choices=["none", "pgd", "fgsm", "r-fgsm", "cw"],
            default="none",
        )
        parser.add_argument(
            "--norm", type=str, choices=["l2", "linf"], default="linf",
        )
        parser.add_argument(
            "--adv-training", action="store_true", default=False
        )
        parser.add_argument(
            "--attack-rate", type=float, default=1.0,
        )
        parser.add_argument(
            "--perturb-target",
            type=str,
            choices=["pos", "neg", "anc"],
            default="pos",
        )
        parser.add_argument(
            "--epsilon", help="l_inf epsilon to use", type=str, default="0.01",
        )
        parser.add_argument(
            "--fgsm-alpha",
            type=str,
            help="Amount of iterations that CW should use",
            default="0.01",
        )
        parser.add_argument(
            "--pgd-steps",
            type=int,
            help="Amount of iterations that PGD should take",
            default="5",
        )
        parser.add_argument(
            "--pgd-alpha", type=str, help="Step size of PGD", default="none",
        )
        parser.add_argument(
            "--pgd-random-restarts",
            type=str,
            help="Amount of random restarts for PGD",
            default="1",
        )
        parser.add_argument(
            "--cw-const",
            type=str,
            help="Const for the CW attack",
            default="1e-1",
        )
        parser.add_argument(
            "--cw-lr",
            type=str,
            help="Learning rate for the CW attack",
            default="8e-4",
        )
        parser.add_argument(
            "--cw-df",
            type=float,
            help="Decrease factor for the CW attack",
            default="0.9",
        )
        parser.add_argument(
            "--cw-iterations",
            type=int,
            help="Amount of iterations that CW should use",
            default="50",
        )
        parser.add_argument(
            "--use-cache-index", action="store_true", default=False
        )

        return parser
