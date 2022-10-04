import os
import tarfile

import pandas as pd

from pkgs.utils import download_file_from_gdrive
from torchvision.datasets import ImageFolder

from .base import BaseSplitImageDataset


class CUB200Dataset(BaseSplitImageDataset):
    def __init__(self, root, **kwargs):
        self.version = (
            "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
        )
        super().__init__(root, **kwargs)

    def fetch(self, folder):
        filename = "CUB_200_2011.tgz"
        raw_file = os.path.join(folder, filename)

        download_file_from_gdrive(
            raw_file,
            url="http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
            # google_id="1LOpRStjqgGzyfYEKuQ69lz_w14xgpwFD",
            file_size=1150585339,
            sha256=self.version,
        )

        extract_folder = os.path.join(folder, "raw")

        if not os.path.exists(extract_folder):
            os.mkdir(extract_folder)

        tar_file = os.path.join(folder, filename)
        with tarfile.open(tar_file, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=extract_folder)

        img_folder = os.path.join(extract_folder, "CUB_200_2011", "images")

        ds = ImageFolder(img_folder)

        return pd.DataFrame(ds.samples).rename(columns={0: "path", 1: "label"})
