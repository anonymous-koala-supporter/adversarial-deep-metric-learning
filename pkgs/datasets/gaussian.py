import os
import tarfile

import pandas as pd

import torch
from pkgs.utils import download_file_from_gdrive
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from .base import BaseSplitImageDataset


class GaussianDataset(BaseSplitImageDataset):
    def __init__(
        self,
        root,
        n_per_class=512,
        dist_a_mean=0.25,
        dist_b_mean=0.75,
        dist_a_std=0.025,
        dist_b_std=0.025,
        **kwargs,
    ):
        self.version = f"{n_per_class}_{dist_a_mean}_{dist_a_std}_{dist_b_mean}_{dist_b_std}"
        self.dist_a_conf = (dist_a_mean, dist_a_std)
        self.dist_b_conf = (dist_b_mean, dist_b_std)
        self.n_per_class = n_per_class

        super().__init__(root, **kwargs)

    def fetch(self, folder):
        base_dim = torch.zeros(3, 224, 224)

        a_mean, a_std = self.dist_a_conf
        a_dist = torch.distributions.normal.Normal(
            base_dim + a_mean, base_dim + a_std
        )

        b_mean, b_std = self.dist_b_conf
        b_dist = torch.distributions.normal.Normal(
            base_dim + b_mean, base_dim + b_std
        )

        extract_folder = os.path.join(folder, "raw")
        if not os.path.exists(extract_folder):
            os.mkdir(extract_folder)

        g_idx = 1
        for c, dist in {"a": a_dist, "b": b_dist}.items():
            class_folder = os.path.join(extract_folder, c)
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)

            for _ in range(self.n_per_class):
                s = dist.sample()
                s = torch.clamp(s, min=0.0, max=1.0)

                path = os.path.join(class_folder, f"{g_idx}.png")
                g_idx += 1

                save_image(s, path)

        ds = ImageFolder(extract_folder)

        return pd.DataFrame(ds.samples).rename(columns={0: "path", 1: "label"})
