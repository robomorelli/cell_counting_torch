#import torch
#from torch import nn
import functools
from fastai.vision.all import *
import fastai

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_layer_name(layer, idx):
    # TODO: minimal implementation based on class name + idx
    # type_str = str(type(layer))
    # type_str = type_str.split('.')[1][:-2]
    # return f"{type_str}_{idx}"

    if isinstance(layer, torch.nn.Conv2d):
        layer_name = 'Conv2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        layer_name = 'ConvT2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.BatchNorm2d):
        layer_name = 'BatchNorm2D_{}_{}'.format(
            idx, layer.num_features)
    elif isinstance(layer, torch.nn.Linear):
        layer_name = 'Linear_{}_{}x{}'.format(
            idx, layer.in_features, layer.out_features
        )
    elif isinstance(layer, fastai.layers.Identity):
        layer_name = 'Identity'
    else:
        layer_name = "Activation_{}".format(idx)
    # idx += 1
    # return layer_name, idx
    return '_'.join(layer_name.split('_')[:2]).lower()


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


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, is_conv=True):
        super(ResidualBlock, self).__init__()

        self.is_conv = is_conv
        self.conv_path = ConvBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add_module('id_path', nn.Conv2d(n_in, n_out, kernel_size=1, padding=0) if self.is_conv else Identity())
        self.id_path = IdentityPath(n_in, n_out, is_conv=is_conv)
        self.add = Add()

    def forward(self, x):
        conv_path = self.conv_path(x)
        short_connect = self.id_path(x)
        return self.add(conv_path, short_connect)


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


class Heatmap(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(Heatmap, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2d(x)) #self.conv2d(x)


class Heatmap2d(nn.Module):
    def __init__(self, n_in, n_out=2, kernel_size=1, stride=1, padding=0, concat_dim=1):
        super(Heatmap2d, self).__init__()
        self.heatmap = Heatmap(n_in, n_out - 1, kernel_size, stride, padding)
        self.concat = Concatenate(dim=concat_dim)

    def forward(self, x):
        heatmap1 = self.heatmap(x)
        heatmap0 = torch.ones_like(heatmap1) - heatmap1
        return self.concat([heatmap0, heatmap1])


# ResUnet
class ResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=1): #switched from n_out=2 (heatmap2D)
        super(ResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        # colorspace transformation
        self.colorspace = nn.Conv2d(3, 1, kernel_size=1, padding=0)

        # block 1
        self.conv_block = ConvBlock(1, n_features_start)
        self.pool1 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 2
        self.residual_block1 = ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True)
        self.pool2 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 3
        self.residual_block2 = ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True)
        self.pool3 = nn.MaxPool2d(pool_ks, pool_stride, pool_pad)

        # block 4: BRIDGE START
        self.c4 = ResidualBlock(4 * n_features_start, 8 *
                                n_features_start, kernel_size=5, padding=2, is_conv=True)

        # block 5: BRIDGE END
        self.c5 = ResidualBlock(8 * n_features_start, 8 *
                                n_features_start, kernel_size=5, padding=2, is_conv=False)

        # block 6
        self.c6 = UpResidualBlock(n_in=8 * n_features_start,
                                  n_out=4 * n_features_start)

        # block 7
        self.c7 = UpResidualBlock(
            4 * n_features_start, 2 * n_features_start)

        # block 8
        self.c8 = UpResidualBlock(
            2 * n_features_start, n_features_start)

        # output
        #self.output = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        self.output = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:

        c0 = self.colorspace(x)
        c1 = self.conv_block(c0)
        p1 = self.pool1(c1)
        c2 = self.residual_block1(p1)
        p2 = self.pool2(c2)
        c3 = self.residual_block2(p2)
        p3 = self.pool3(c3)
        c4 = self.c4(p3)
        c5 = self.c5(c4)
        c6 = self.c6(c5, c3)
        c7 = self.c7(c6, c2)
        c8 = self.c8(c7, c1)
        output = self.output(c8)

        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
