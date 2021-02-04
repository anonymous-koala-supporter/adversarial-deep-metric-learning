import glob
import json
import math
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import PIL

import torch
from skimage.transform import resize
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def split_classes_isolated(labels, ratio=None, max_abs=float("inf")):
    label_counts = np.array(np.unique(labels, return_counts=True))

    max_other_size = np.min([(1 - ratio) * len(labels), max_abs])
    other_size = 0
    other_labels = []

    while True:
        candidate_labels = label_counts[1, :] <= (max_other_size - other_size)

        if candidate_labels.sum() == 0:
            # no available candidates
            break

        prob = candidate_labels / candidate_labels.sum()
        idx = np.random.choice(len(candidate_labels), p=prob)

        other_size += label_counts[1, idx]

        other_labels.append(label_counts[0, idx])
        label_counts = np.delete(label_counts, idx, 1)

    return label_counts[0, :], np.array(other_labels)


def create_splits_df(
    df: pd.DataFrame, n_splits=None, train_ratio=None, **kwargs
) -> pd.DataFrame:
    for i in range(1, n_splits + 1):
        tr_labels, _ = split_classes_isolated(
            df["label"].to_numpy(), ratio=train_ratio
        )
        df[f"split_isolated_{i}"] = df["label"].isin(tr_labels).astype(int)

    for i in range(1, n_splits + 1):
        tr_idx = np.random.rand(len(df)) < train_ratio
        df[f"split_random_{i}"] = tr_idx.astype(int)

    return df


class BaseSplitImageDataset(Dataset):
    def __init__(
        self,
        root,
        split=1,
        split_kind=None,
        train=True,
        n_splits=5,
        train_ratio=0.5,
        transform=None,
        return_idx=False,
        loader=None,
    ):
        self.root = root
        self.return_idx = return_idx

        conf = {
            "version": self.get_version(),
            "n_splits": n_splits,
            "train_ratio": train_ratio,
        }

        df = None
        csv_file = os.path.join(root, "data_splits.csv")

        self.loader = default_loader if loader is None else loader

        has_files, has_splits = self.verify_local_variant(conf)
        if not has_files:
            df = self.__init_image_data()
            df.to_csv(csv_file, index=False)
            self.set_config_file({"version": self.get_version()})

        if df is None:
            df = pd.read_csv(csv_file)

        if not has_splits:
            df = create_splits_df(df, **conf)
            df.to_csv(csv_file, index=False)
            self.set_config_file(conf)

        if split == "sorted":
            uniq_labels = df["label"].unique()
            cut = round(len(uniq_labels) * train_ratio)
            labels_for_set = uniq_labels[:cut] if train else uniq_labels[cut:]
            idx = df["label"].isin(labels_for_set)
        elif isinstance(split, (str, int)) and str(split).isdigit():
            group = 1 if train else 0
            idx = df[f"split_{split_kind}_{split}"] == group
        else:
            group = 1 if train else 0
            idx = df[f"split_{split}"] == group

        labels = df["label"].unique()
        labels.sort()
        m = {labels[i]: i for i in range(len(labels))}
        df["label"] = df["label"].apply(lambda x: m[x])

        df = df[idx][["path", "label"]]

        self.samples = df.to_numpy()
        self.root = root

        if transform is None:
            if train:
                transform = [
                    transforms.RandomResizedCrop(
                        size=224, scale=(0.6, 1.0)
                    ),  # Roth et al. did not adjust scale, however, defaults range is crazy (0.08, 1.0)
                    transforms.RandomHorizontalFlip(0.5),
                ]
            else:
                transform = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]

            transform.append(transforms.ToTensor())
            transform = transforms.Compose(transform)

        self.transform = transform

    def __init_image_data(self) -> pd.DataFrame:
        self.__empty_root_folder()
        df = self.fetch(self.root)

        for k in ["path", "label"]:
            assert k in df.keys(), f"DataFrame must have column: {k}"

        tmp_imgs_folder = os.path.join(self.root, "__images")
        new_paths = []
        for _, row in df.iterrows():
            path, label = row["path"], row["label"]
            class_folder = os.path.join(tmp_imgs_folder, str(label))
            Path(class_folder).mkdir(parents=True, exist_ok=True)

            filename = Path(path).name
            dst_path = os.path.join(class_folder, filename)

            shutil.move(path, dst_path)
            new_path = os.path.join("images", str(label), filename)

            new_paths.append(new_path)

        df["path"] = new_paths

        for f in glob.glob(os.path.join(self.root, "*")):
            if not f.endswith("__images"):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                    continue

                os.remove(f)

        imgs_folder = os.path.join(self.root, "images")
        shutil.move(tmp_imgs_folder, imgs_folder)

        return df

    def __empty_root_folder(self):
        if os.path.isdir(self.root):
            shutil.rmtree(self.root)

        Path(self.root).mkdir(parents=True, exist_ok=True)

    def __config_file(self) -> str:
        return os.path.join(self.root, ".config.json")

    def set_config_file(self, conf):
        with open(self.__config_file(), "w") as f:
            f.write(json.dumps(conf))

    def get_version(self) -> str:
        return self.version

    def verify_local_variant(self, conf) -> Tuple[bool, bool]:
        if not os.path.isfile(self.__config_file()):
            return False, False

        try:
            with open(self.__config_file(), "r") as f:
                cache_conf = json.load(f)

            if cache_conf["version"] != conf["version"]:
                raise ValueError("incorrect version")

        except Exception:
            return False, False

        for k, v in conf.items():
            if not k in cache_conf:
                return True, False

            if cache_conf[k] != v:
                return True, False

        return True, True

    def fetch(self, folder: str) -> pd.DataFrame:
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_file, y = self.samples[idx, :2]
        img_name = os.path.join(self.root, im_file)
        sample = self.loader(img_name)

        if self.transform:
            sample = self.transform(sample)

        if self.return_idx:
            return sample, y, idx

        return sample, y

    def get_labels(self):
        return torch.from_numpy(self.samples[:, 1].astype(int))


class ClassBalancedSampler(Sampler):
    def __init__(
        self, classes, batch_size, m_per_class=40, strict=False, resample=True,
    ):
        assert (batch_size % m_per_class) == 0

        if strict:
            assert (
                len(classes.unique()) * m_per_class >= batch_size
            ), f"too few classes (= {len(classes.unique())})"

        self.classes = classes
        self.batch_size = batch_size
        self.m_per_class = m_per_class
        self.strict = strict
        self.resample = resample

        self.batches, self.n_batches = self._batches()

    def _batches(self):
        classes, batch_size, m_per_class = (
            self.classes,
            self.batch_size,
            self.m_per_class,
        )

        ul, cts = classes.unique(return_counts=True)
        idxs = [
            torch.where(classes == c)[0][torch.randperm(n)]
            for c, n in zip(ul, cts)
        ]
        groups_per_batch = int(batch_size / m_per_class)

        batches = []
        while cts.sum() > 0:
            _cts = cts.float()
            if self.strict:
                _cts[_cts < m_per_class] = 0

                if _cts.sum() == 0:
                    break

            p = _cts / _cts.sum()
            s = torch.multinomial(p, min(groups_per_batch, (p > 0).sum()))

            select_cts = cts[s]
            select_cts[select_cts >= m_per_class] = m_per_class

            batch = torch.cat(
                [idxs[c][-cts[c] :][: select_cts[i]] for i, c in enumerate(s)]
            )
            batches.append(batch)

            cts[s] -= select_cts

        if self.strict:
            batches = [b for b in batches if len(b) == batch_size]
            n = len(batches)
            return torch.cat(batches), n

        return torch.cat(batches), math.ceil(len(classes) / batch_size)

    def __iter__(self):
        if self.resample:
            self.batches, self.n_batches = self._batches()

        return iter(
            [
                self.batches[i * self.batch_size : (i + 1) * self.batch_size]
                for i in range(self.n_batches)
            ]
        )

    def __len__(self):
        return self.n_batches


class SKResize(object):
    def __init__(self, size, anti_aliasing=False):
        self.size = size
        self.anti_aliasing = anti_aliasing

    def __call__(self, im):
        if isinstance(im, PIL.Image.Image):
            im = np.asarray(im)[:, :, :3] / np.array(255, dtype=np.float32)

        im = resize(im, self.size, anti_aliasing=self.anti_aliasing)

        return im
