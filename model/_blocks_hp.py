#  #!/usr/bin/env python3
#  -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Luca Clissa
#  #Licensed under the Apache License, Version 2.0 (the "License");
#  #you may not use this file except in compliance with the License.
#  #You may obtain a copy of the License at
#  #http://www.apache.org/licenses/LICENSE-2.0
#  #Unless required by applicable law or agreed to in writing, software
#  #distributed under the License is distributed on an "AS IS" BASIS,
#  #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  #See the License for the specific language governing permissions and
#  #limitations under the License.
__all__ = ['_get_ltype', 'Add', 'Concatenate', 'ConvBlock', 'ResidualBlock', 'UpResidualBlock', 'Bottleneck',
           'Heatmap', 'Heatmap2d', 'UpResidualBlockVAE_old', 'BottleneckVAE' , 'HeatmapVAE', 'HeatmapVAERecon',
           'UpResidualBlockNoConcat', 'UpResidualBlockNoConv']

from fastai.vision.all import *
from ._utils import *
from model.utils import InverseSquareRootLinearUnit, Dec1, ConstrainedConv2d


# Utils
def _get_ltype(layer):
    name = str(layer.__class__).split("'")[1]
    return name.split('.')[-1]


# Blocks
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.add

    def forward(self, x1, x2):
        return self.add(x1, x2)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.cat = partial(torch.cat, dim=dim)

    def forward(self, x):
        return self.cat(x)


# class ConvBlock(nn.Module):
#     def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.BatchNorm2d(n_in, momentum=0.01, eps=0.001),
#             nn.ELU(),
#             nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
#             nn.BatchNorm2d(n_out, momentum=0.01, eps=0.001),
#             nn.ELU(),
#             nn.Conv2d(n_out, n_out, kernel_size, stride, padding),
#         )
#
#     def forward(self, x):
#         return self.block(x)

# TODO: commented parts should allow blocks numbering. Find a way to do it automatically and propagate to different blocks
class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):  # , idb=1):
        super(ConvBlock, self).__init__()

        layers = [
            nn.BatchNorm2d(n_in, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_out, n_out, kernel_size, stride, padding),
        ]
        # self.idb = idb
        self._init_block(layers)

    def forward(self, x):
        for layer in self.conv_block.values():
            x = layer(x)
        return x

    def _init_block(self, layers):
        # self.add_module(f"conv_block{self.idb}", nn.ModuleDict())
        self.conv_block = nn.ModuleDict()
        for idx, layer in enumerate(layers):
            self.conv_block.add_module(get_layer_name(layer, idx), layer)
            # getattr(self, f"conv_block{self.idb}")[get_layer_name(layer, idx)] = layer


class IdentityPath(nn.Module):
    def __init__(self, n_in: int, n_out: int, is_conv: bool = True, upsample: bool = False):
        super(IdentityPath, self).__init__()

        self.is_conv = is_conv
        self.upsample = upsample

        # TODO:
        #  1) find elegant way to deal with is_conv=False, upsample=True; currently returns ConvT2d
        #  2) implement up_conv + concatenate directly here
        if upsample:
            layer = nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0)
        elif is_conv:
            layer = nn.Conv2d(n_in, n_out, kernel_size=1, padding=0)
        else:
            layer = Identity()
        self.layer_name = get_layer_name(layer, 1)
        self.add_module(self.layer_name, layer)

    def forward(self, x):
        return getattr(self, self.layer_name)(x)


# class ConvResNetBlock(nn.Module):
#     def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
#         super(ConvResNetBlock, self).__init__()
#         self.conv_block = ConvBlock(
#             n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.short_connect = nn.Conv2d(n_in, n_out, kernel_size=1, padding=0)
#         self.resnet_block = Add()
#
#     def forward(self, x):
#         conv_block = self.conv_block(x)
#         short_connect = self.short_connect(x)
#         resnet_block = self.resnet_block(conv_block, short_connect)
#         return resnet_block
#
#
# class ResNetBlock(nn.Module):
#     def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
#         super(ResNetBlock, self).__init__()
#         self.conv_block = ConvBlock(
#             n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.resnet_block = Add()
#
#     def forward(self, x1, x2):
#         conv_block = self.conv_block(x1)
#         resnet_block = self.resnet_block(conv_block, x2)
#         return resnet_block


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, is_conv=True):
        super(ResidualBlock, self).__init__()

        self.is_conv = is_conv
        self.conv_path = ConvBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.id_path = IdentityPath(n_in, n_out, is_conv=is_conv)
        self.add = Add()

    def forward(self, x):
        conv_path = self.conv_path(x)
        short_connect = self.id_path(x)
        return self.add(conv_path, short_connect)


# class UpResNetBlock(nn.Module):
#     def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
#         super(UpResNetBlock, self).__init__()
#         self.up_conv = nn.ConvTranspose2d(
#             n_in, n_out, kernel_size=2, stride=2, padding=0)
#         self.concat = Concatenate(dim=concat_dim)
#         self.conv_block = ConvBlock(
#             n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.up_resnet_block = Add()
#
#     def forward(self, x, long_connect):
#         short_connect = self.up_conv(x)
#         concat = self.concat([short_connect, long_connect])
#         up_resnet_block = self.up_resnet_block(
#             self.conv_block(concat), short_connect)
#         return up_resnet_block


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, stride=1, padding=2):
        super(Bottleneck, self).__init__()

        self.residual_block1 = ResidualBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=True)
        self.residual_block2 = ResidualBlock(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=False)

    def forward(self, x):
        return self.residual_block2(self.residual_block1(x))



class BottleneckVAE(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, stride=1,
                 padding=2, featureDim=128*64*64, zDim=64):
        super(BottleneckVAE, self).__init__()

        self.residual_block1 = ResidualBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=True)
        self.residual_block2 = ResidualBlock(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=False)

        self.encFC1 = ConstrainedConv2d(n_out, 1, 1, stride, padding=0)
        self.encFC2 = ConstrainedConv2d(n_out, 1, 1, stride, padding=0)

        #self.encFC1 = nn.Linear(featureDim, zDim)
        #self.encFC2 = nn.Linear(featureDim, zDim)
        self.act_sigma = InverseSquareRootLinearUnit()
        self.featureDim = featureDim

    def forward(self, x):
        x = self.residual_block2(self.residual_block1(x))
        #x = x.view(-1, self.featureDim)
        return self.encFC1(x), self.act_sigma(self.encFC2(x)), x

#class BottleneckVAE(nn.Module):
        #    def __init__(self, n_in, n_out, kernel_size=5, stride=1, padding=2):
        #        super(BottleneckVAE, self).__init__()
        #
        #        self.conv2d_mu = nn.Conv2d(n_in, 1, 1, stride, padding=0)
    #        self.conv2d_sigma = nn.Conv2d(n_in, 1, 1, stride, padding=0)
    #
        #    def forward(self, x):
#        return self.conv2d_mu(x), nn.ReLU()((self.conv2d_sigma(x)))
        #return self.conv2d_mu(self.residual_block2(self.residual_block1(x))), self.conv2d_sigma(self.residual_block2(self.residual_block1(x)))


class UpResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlock, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
            "concat": Concatenate(dim=concat_dim)
        })
        self.conv_path = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = Add()

    def forward(self, x, long_connect):
        short_connect = self.id_path.up_conv(x)
        concat = self.id_path.concat([short_connect, long_connect])
        return self.add(self.conv_path(concat), short_connect)

class UpResidualBlockNoConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlockNoConv, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
            "concat": Concatenate(dim=concat_dim)
        })
        #self.conv_path = ConvBlock(
        #    n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.add = Add()

    def forward(self, x, long_connect):
        short_connect = self.id_path.up_conv(x)
        concat = self.id_path.concat([short_connect, long_connect])
        return concat

class UpResidualBlockNoConcat(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlockNoConcat, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
           # "concat": Concatenate(dim=concat_dim)
        })
        self.conv_path = ConvBlock(
            n_in//2, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = Add()

    def forward(self, x):
        short_connect = self.id_path.up_conv(x)
        #concat = self.id_path.concat([short_connect, long_connect])
        return self.add(self.conv_path(short_connect), short_connect)

class UpResidualBlockVAE_old(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlockVAE, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
            "concat": Concatenate(dim=concat_dim)
        })
        self.conv_path = ConvBlock(
            n_in*2, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = Add()

    def forward(self, x, long_connect):
        short_connect = self.id_path.up_conv(x)
        concat = self.id_path.concat([short_connect, long_connect])
        return self.add(self.conv_path(concat), short_connect)

class Heatmap(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(Heatmap, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2d(x))

class HeatmapVAE(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(HeatmapVAE, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        #self.ConstrainedConv2d_mu = ConstrainedConv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2d(x))

class HeatmapVAERecon(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(HeatmapVAERecon, self).__init__()
        self.ConstrainedConv2d_mu = ConstrainedConv2d(n_in, n_out, kernel_size, stride, padding)
        self.ConstrainedConv2d_sigma = ConstrainedConv2d(n_in, n_out, kernel_size, stride, padding)
        #self.conv2d_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        #self.conv2d_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act_mu = nn.Sigmoid()
        self.act_sigma = InverseSquareRootLinearUnit() #self.act_sigma = nn.Softplus()

    def forward(self, x):
        return self.act_mu(self.ConstrainedConv2d_mu(x)), self.act_sigma(self.ConstrainedConv2d_sigma(x))

class Heatmap2d(nn.Module):
    def __init__(self, n_in, n_out=2, kernel_size=1, stride=1, padding=0, concat_dim=1):
        super(Heatmap2d, self).__init__()
        self.heatmap = Heatmap(n_in, n_out - 1, kernel_size, stride, padding)
        self.concat = Concatenate(dim=concat_dim)

    def forward(self, x):
        heatmap1 = self.heatmap(x)
        heatmap0 = torch.ones_like(heatmap1) - heatmap1
        return self.concat([heatmap0, heatmap1])