# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is an aggregated form of software package found here:
#   https://github.com/Confusezius/Deep-Metric-Learning-Baselines/tree/cd484d92a362a6fbfcfdcddd66b63026bdd28fda


import pretrainedmodels as ptm
import torch.nn as nn
from torchvision.models import inception


def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.
    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


class GoogLeNet(nn.Module):
    """
    Container for GoogLeNet s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, embedding_size, pretrained=False, normalize_last=True):
        """
        Args:
            opt: argparse.Namespace, contains all training-specific parameters.
        Returns:
            Nothing!
        """
        super(GoogLeNet, self).__init__()

        self.normalize_last = normalize_last

        self.model = inception(
            num_classes=1000, pretrained="imagenet" if pretrained else False,
        )

        for module in filter(
            lambda m: type(m) == nn.BatchNorm2d, self.model.modules()
        ):
            module.eval()
            module.train = lambda _: None

        rename_attr(self.model, "fc", "last_linear")

        self.layer_blocks = nn.ModuleList(
            [
                self.model.inception3a,
                self.model.inception3b,
                self.model.maxpool3,
                self.model.inception4a,
                self.model.inception4b,
                self.model.inception4c,
                self.model.inception4d,
                self.model.inception4e,
                self.model.maxpool4,
                self.model.inception5a,
                self.model.inception5b,
                self.model.avgpool,
            ]
        )

        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features, embedding_size
        )

    def forward(self, x):
        x = self.model.conv3(
            self.model.conv2(self.model.maxpool1(self.model.conv1(x)))
        )
        x = self.model.maxpool2(x)

        ### Inception Blocks
        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = x.view(x.size(0), -1)
        x = self.model.dropout(x)

        x = self.model.last_linear(x)
        if not self.normalize_last:
            return x

        x = nn.functional.normalize(x, dim=-1)

        return x


class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, embedding_size, pretrained=True, normalize_last=True):
        super(ResNet50, self).__init__()
        self.normalize_last = normalize_last

        self.model = ptm.__dict__["resnet50"](
            num_classes=1000, pretrained="imagenet" if pretrained else None
        )

        for module in filter(
            lambda m: type(m) == nn.BatchNorm2d, self.model.modules()
        ):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features, embedding_size
        )

        self.layer_blocks = nn.ModuleList(
            [
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ]
        )

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(
            self.model.relu(self.model.bn1(self.model.conv1(x)))
        )

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.model.last_linear(x)

        if not self.normalize_last:
            return x

        x = nn.functional.normalize(x, dim=-1)

        return x
