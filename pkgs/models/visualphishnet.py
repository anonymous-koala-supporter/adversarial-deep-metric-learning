import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vgg16


class VisualPhishNet(nn.Module):
    def __init__(self, embedding_size, pretrained=False, normalize_last=False):
        super(VisualPhishNet, self).__init__()
        _vgg16 = vgg16(pretrained=False, init_weights=False)

        modules = list(_vgg16.features) + [
            nn.Conv2d(512, embedding_size, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        ]

        self.normalize_last = normalize_last
        self.embedding_size = embedding_size
        self.layers = nn.Sequential(*modules)

        if pretrained is not False:
            self.load_org_weights(pretrained)

    def load_org_weights(self, variant):
        if not isinstance(variant, str):
            raise ValueError(
                f"incorrect value for pretrained ({variant}), must be: natural, benign, or <path to weights>"
            )

        abspath = Path(__file__).parent.absolute()
        weight_path = ""
        if variant.endswith(".pt"):
            weight_path = variant
        elif variant == "robust":
            weight_path = os.path.join(
                abspath, "..", "..", "assets", "models", "vpnet_robust.pt"
            )
        elif variant == "natural":
            weight_path = os.path.join(
                abspath, "..", "..", "assets", "models", "vpnet_natural.pt"
            )
        else:
            raise ValueError(f"unknown variant: '{variant}'")

        self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        x = self.layers(x).view(-1, self.embedding_size)

        if self.normalize_last:
            x = nn.functional.normalize(x, dim=-1)

        return x
