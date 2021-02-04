import os
import zipfile

import pandas as pd

from torchvision.datasets.utils import download_url

from .base import BaseSplitImageDataset


class SOPDataset(BaseSplitImageDataset):
    def __init__(self, root, **kwargs):
        self.version = "7f73d41a2f44250d4779881525aea32e"
        super().__init__(root, **kwargs)

    def fetch(self, folder):
        filename = "Stanford_Online_Products.zip"

        extract_folder = os.path.join(folder, "raw")
        url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
        md5 = self.version

        download_url(url, folder, filename, md5)

        if not os.path.exists(extract_folder):
            os.mkdir(extract_folder)

        zip_file = os.path.join(folder, filename)
        with zipfile.ZipFile(zip_file, "r") as zipf:
            zipf.extractall(path=extract_folder)

        org_img_folder = os.path.join(
            extract_folder, "Stanford_Online_Products"
        )

        classes = [
            "bicycle",
            "cabinet",
            "chair",
            "coffee_maker",
            "fan",
            "kettle",
            "mug",
            "lamp",
            "sofa",
            "stapler",
            "table",
            "toaster",
        ]

        df = pd.DataFrame()
        for c in classes:
            class_folder = os.path.join(org_img_folder, f"{c}_final")

            img_files = []
            labels = []
            for f in os.listdir(class_folder):
                p = os.path.join(class_folder, f)
                if not os.path.isfile(p):
                    continue

                basename, _ = os.path.splitext(f)
                prod_id, _ = basename.split("_")

                img_files.append(p)
                labels.append(prod_id)

            df = df.append(
                pd.DataFrame({"path": img_files, "label": labels}),
                ignore_index=True,
            )

        return df
