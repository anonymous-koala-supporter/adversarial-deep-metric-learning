import os
import tarfile

import pandas as pd
import scipy.io

from torchvision.datasets.utils import download_url

from .base import BaseSplitImageDataset


class CARS196Dataset(BaseSplitImageDataset):
    def __init__(self, root, **kwargs):
        self.version = "d5c8f0aa497503f355e17dc7886c3f14"
        super().__init__(root, **kwargs)

    def fetch(self, folder):
        name = "car_ims"
        url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
        md5 = self.version
        filename = f"{name}.tgz"

        download_url(url, folder, filename, md5)
        extract_folder = os.path.join(folder, f"{filename}_raw")

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

        org_img_folder = os.path.join(extract_folder, name)

        mat_file = os.path.join(folder, "cars_annos.mat")
        url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"
        md5 = "b407c6086d669747186bd1d764ff9dbc"

        download_url(url, folder, "cars_annos.mat", md5)

        mat = scipy.io.loadmat(mat_file)
        out = [
            (row[0][0].split(os.sep)[-1], row[5][0][0])
            for row in mat["annotations"][0]
        ]
        df = pd.DataFrame(out, columns=["path", "label"])
        df["path"] = df["path"].apply(
            lambda x: os.path.join(org_img_folder, x)
        )

        return df
