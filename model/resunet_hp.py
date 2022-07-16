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
__all__ = ['ResUnet', 'c_resunet', 'ResUnetAE', 'c_resunetAE', 'ResUnetVAE', 'c_resunetVAE', '_resunetVAE']

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
from model.utils import InverseSquareRootLinearUnit, Dec1, ConstrainedConv2d
#from fluocells.config import MODELS_PATH


class ResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=1):
        super(ResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
        })

        self.decoder = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2 * n_features_start, n_features_start),
        })

        # output
        #self.head = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        self.head = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if 'block' in lbl: downblocks.append(x)
            # NEXT loop is hon the values and so we don't hane the name as in the items of the previous loop
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resunet(
        arch: str,
        n_features_start: int,
        n_out: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnet(n_features_start, n_out)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        weights_path = MODELS_PATH / f"{arch}_state_dict.pkl"
        print('loading pretrained Keras weights from', weights_path)
        keras_weights = load_pkl(weights_path)
        keras_state_dict = pt2k_state_dict(model.state_dict())
        assert len(keras_weights) == len(keras_state_dict)
        transfer_weights(model, keras_weights, keras_state_dict)
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #         model.load_state_dict(state_dict)
    else:
        model.init_kaiming_normal()
    return model

class ResUnetAE(nn.Module):
    def __init__(self, n_features_start=16, n_out=3):
        super(ResUnetAE, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0


        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
        })

        self.decoder = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2 * n_features_start, n_features_start),
        })

        # output
        #self.head = Heatmap2d(
        #    n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        self.head = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if 'block' in lbl: downblocks.append(x)
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resunetAE(
        arch: str,
        n_features_start: int,
        n_out: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnetAE(n_features_start, n_out)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        weights_path = MODELS_PATH / f"{arch}_state_dict.pkl"
        print('loading pretrained Keras weights from', weights_path)
        keras_weights = load_pkl(weights_path)
        keras_state_dict = pt2k_state_dict(model.state_dict())
        assert len(keras_weights) == len(keras_state_dict)
        transfer_weights(model, keras_weights, keras_state_dict)
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #         model.load_state_dict(state_dict)
    else:
        model.init_kaiming_normal()
    return model

class ResUnetVAE(nn.Module):
    def __init__(self, n_features_start=16, zDim=64, n_out=1, n_outRec=3):
        super(ResUnetVAE, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.act2 = InverseSquareRootLinearUnit()
        self.n_out = n_out
        self.n_outRec = n_outRec
        self.zDim = zDim
        self.n_features_start = n_features_start

        self.encoder = nn.ModuleDict({
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            #'bottleneck': BottleneckVAE(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2,
            #                            featureDim=8 * self.n_features_start*64*64, zDim=64),
            'bottleneck': BottleneckVAE(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2,
                                        featureDim=4 * self.n_features_start * 64 * 64, zDim=zDim),
        })

        #self.pre_up = Dec1(64, 128 * 64 * 64) # switch to (64, 128*64*64)>>>upconv_block:UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start)
        #self.pre_up = Dec1(zDim, 64 * 64 * 64)
        self.pre_up  = nn.Conv2d(1, 8 * n_features_start, 1, 1, padding=0)
        self.rebase_up = nn.ConvTranspose2d(n_features_start, 1, kernel_size=2, stride=2, padding=0)


        self.decoder_segm = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2*n_features_start, n_features_start),
        })

        self.decoder_rec = nn.ModuleDict({
            # block 6
            'upconv_block1NoConcat': UpResidualBlockNoConcat(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2NoConcat': UpResidualBlockNoConcat(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3NoConcat': UpResidualBlockNoConcat(2*n_features_start, n_features_start),
        })

        self.decoder_conc = nn.ModuleDict({
            # block 6
            'upconv_block1NoConv': UpResidualBlockNoConv(n_in=8 * n_features_start, n_out=4 * n_features_start),
            #'upconv_block1': UpResidualBlockVAE(n_in=4 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2NoConvt': UpResidualBlockNoConv(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3NoConv': UpResidualBlockNoConv(2*n_features_start, n_features_start),
        })


        self.headSeg = HeatmapVAE(self.n_features_start, self.n_out, kernel_size=1, stride=1, padding=0)
        #self.headConc = HeatmapVAE(n_features_start, self.n_out, kernel_size=1, stride=1, padding=0)
        self.headRec = HeatmapVAERecon(self.n_features_start, self.n_outRec, kernel_size=1, stride=1, padding=0)

    def reparameterize_logvar(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def reparameterize(self, mu, sigma):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = sigma # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            if "bottle" in lbl:
                mu, sigma, x_bottle = layer(x)
                sigma = self.act2(sigma)
            else:
                x = layer(x)
                if "color" in lbl:
                    gray_rgb = x
                if "pool1" in lbl:
                    recon_base = x
                if 'block' in lbl: downblocks.append(x)

        # after bottleneck x came back to (bs,64,64,64) size becauese the view is used inside the bottnk block
        z = self.reparameterize(mu, sigma)
        z = self.pre_up(z)
        #x = nn.ReLU()(x) # TO REMOVE
        #x = x.view(-1, self.n_features_start * 4, 64, 64) #x = x.view(-1, self.n_features_start*8, 64, 64)

        x_seg = x_bottle
        x_conc = x_bottle
        # PATH FOR SEGMENTATION
        for layer, long_connect in zip(self.decoder_segm.values(), reversed(downblocks)):
            x_seg = layer(x_seg, long_connect)
        segm_out = self.headSeg(x_seg)

        # PATH FOR CONCATENATION
        #for layer, long_connect in zip(self.decoder_conc.values(), reversed(downblocks)):
        #    x_conc = layer(x_conc, long_connect)
        #conc_out = self.headConc(x_conc)
        conc_out = self.rebase_up(recon_base)
        #conc_out = self.headConc(x_conc)

        # PATH FOR RECONSTRUCTION
        for lbl, layer in self.decoder_rec.items(): #downblock store the long connection
            z = layer(z)
        recon_out = self.headRec(z)

        #return mu, sigma, segm_out, recon_out
        return mu, sigma, segm_out, conc_out, recon_out

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resunetVAE(
        arch: str,
        n_features_start: int,
        zDim: int,
        n_out: int,
        n_outRec: int,
        #     block: Type[Union[BasicBlock, Bottleneck]],
        #     layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs,
) -> ResUnet:
    model = ResUnetVAE(n_features_start, zDim , n_out,  n_outRec)  # , **kwargs)
    model.__name__ = arch
    # TODO: implement weights fetching if not present
    if pretrained:
        weights_path = MODELS_PATH / f"{arch}_state_dict.pkl"
        print('loading pretrained Keras weights from', weights_path)
        keras_weights = load_pkl(weights_path)
        keras_state_dict = pt2k_state_dict(model.state_dict())
        assert len(keras_weights) == len(keras_state_dict)
        transfer_weights(model, keras_weights, keras_state_dict)
    #         state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #         model.load_state_dict(state_dict)
    else:
        model.init_kaiming_normal()
    return model


def c_resunet(arch='c-ResUnet', n_features_start: int = 16, n_out: int = 1, pretrained: bool = False,
              progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunet(arch=arch, n_features_start=n_features_start, n_out=n_out, pretrained=pretrained,
                    progress=progress, **kwargs)

def c_resunetAE(arch='c-ResUnetAE', n_features_start: int = 16, n_out: int = 3, pretrained: bool = False,
              progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunetAE(arch=arch, n_features_start=n_features_start, n_out=n_out, pretrained=pretrained,
                    progress=progress, **kwargs)

def c_resunetVAE(arch='c-ResUnetVAE', n_features_start: int = 16, zDim: int = 64, n_out: int = 3,  n_outRec=3,
                 pretrained: bool = False, progress: bool = True,
              **kwargs) -> ResUnet:
    r"""cResUnet model from `"Automating Cell Counting in Fluorescent Microscopy through Deep Learning with c-ResUnet"
    <https://www.nature.com/articles/s41598-021-01929-5>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resunetVAE(arch=arch, n_features_start=n_features_start, zDim = zDim, n_out=n_out,  n_outRec=n_outRec,
                       pretrained=pretrained, progress=progress, **kwargs)
