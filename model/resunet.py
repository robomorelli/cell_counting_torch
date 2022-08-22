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
__all__ = ['ResUnet', 'c_resunet', 'load_model', 'load_ae_inference']

from fastai.vision.all import *
from ._blocks import *
from ._utils import *
from model.utils import InverseSquareRootLinearUnit, Dec1, ConstrainedConv2d, LinConstr
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
        if n_out > 1:
            self.head = Heatmap(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)
        else:
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
        print('error')
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


def load_model(resume_path, device, n_features_start=16, n_out=1, fine_tuning=False
               , unfreezed_layers=1):

    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=n_out,
                                      device=device).to(device))
    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        if unfreezed_layers.isdecimal():
            unfreezed_layers = int(unfreezed_layers)
        for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
            #print('unfreezing {} of {}'.format(unfreezed_layers, block))
            if block[0] == 'head':
                for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                    if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                            #print(block, n, p.requires_grad)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                            #print(block, n, p.requires_grad)
                unfreezed_layers = int(unfreezed_layers) - 1

            else:
                for nc, cc in list(block[1].named_children())[::-1]:
                    if unfreezed_layers > 0:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                        print('keep freezed {}'.format(nc))
                    unfreezed_layers = int(unfreezed_layers) - 1

        print('requires grad for each layer:')
        for block in list(list(model.children())[0].named_children())[::-1]:
            for n, p in list(block[1].named_parameters()[::-1]):
                print(n, p.requires_grad)

    return model

def load_ae_inference(resume_path, device, n_features_start=16, n_out=3, fine_tuning=False, unfreezed_layers=1):
    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=n_features_start, n_out=n_out,
                  device=device).to(device))

    layers_to_remove = ['module.head.conv2d.weight', 'module.head.conv2d.bias']
    layers_to_rename = ['module.head.conv2d_binary.weight', 'module.head.conv2d_binary.bias'] #take automatically the name (they are the last lauyers)
    checkpoint_file = torch.load(resume_path)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_remove:
            checkpoint_file.pop(k)
    for k in list(checkpoint_file.keys()):
        if k in layers_to_rename:
            checkpoint_file[k.replace("_binary", "")] = checkpoint_file.pop(k)

    model.load_state_dict(checkpoint_file, strict=False)
    if fine_tuning:
        print('fine_tuning')
        if unfreezed_layers.isdecimal():
            unfreezed_layers = int(unfreezed_layers)
        for block in list(list(model.children())[0].named_children())[::-1]:  # encoder, decoder, head
            #print('unfreezing {} of {}'.format(unfreezed_layers, block))
            if block[0] == 'head':
                for nc, cc in list(block[1].named_children())[::-1]:  # [1] because 0 is the name
                    if unfreezed_layers > 0: #and isinstance(c, nn.Conv2d):
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                            #print(block, n, p.requires_grad)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                            #print(block, n, p.requires_grad)
                unfreezed_layers = int(unfreezed_layers) - 1

            else:
                for nc, cc in list(block[1].named_children())[::-1]:
                    if unfreezed_layers > 0:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(True)
                        print('unfreezed {}'.format(nc))
                    else:
                        for n, p in cc.named_parameters():
                            p.requires_grad_(False)
                        print('keep freezed {}'.format(nc))
                    unfreezed_layers = int(unfreezed_layers) - 1

        print('requires grad for each layer:')
        for block in list(list(model.children())[0].named_children())[::-1]:
            for n, p in list(block[1].named_parameters())[::-1]:
                print(n, p.requires_grad)

    return model


#for nnns, cccs in list(block[1].named_children())[::-1]:
    #    for nns, ccs in list(cccs.named_children()):
    #    for ns, cs in list(ccs.named_children()):
    #        for n, c in list(cs.named_children()):
    #            if unfreezed_layers > 0:  # and (isinstance(c, nn.Conv2d) or isinstance(c, nn.ConvTranspose2d)):
    #                for n, p in c.named_parameters():
    #                    p.requires_grad_(True)
    #            else:
    #                for n, p in c.named_parameters():
    #                    p.requires_grad_(False)
    #unfreezed_layers = int(unfreezed_layers) - 1