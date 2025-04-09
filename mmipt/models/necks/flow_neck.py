# Copyright (c) MMIPT. All rights reserved.
from math import ceil
from typing import Dict

import torch
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule, ModuleList, normal_init
from mmengine.utils import is_seq_of
from torch import Tensor, nn
from torch.nn import functional as F

from mmipt.registry import MODELS
from mmipt.utils import MultiTensor


@MODELS.register_module()
class DefaultFlow(BaseModule):

    def __init__(
        self,
        in_channels,
        out_channels=3,
        kernel_size=3,
        bias=True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=None)

        self.conv = build_conv_layer(
            cfg=dict(type=f'Conv{out_channels}d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.conv(x)
        return x

    def init_weights(self):
        normal_init(self.conv, mean=0, std=1e-5, bias=0)


@MODELS.register_module()
class ResizeFlow(BaseModule):

    def __init__(
            self,
            img_size,
            in_channels,
            resize_channels=(32, 32),
            cps=(3, 3, 3),
            init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        ndim = len(img_size)

        # determine and set output control point sizes from image size and control point spacing
        for i, c in enumerate(cps):
            if c > 8 or c < 2:
                raise ValueError(f'Control point spacing ({c}) at dim ({i}) '
                                 f'not supported, must be within [1, 8]')

        self.output_size = tuple([
            int(ceil((imsz - 1) / c) + 1 + 2)
            for imsz, c in zip(img_size, cps)
        ])

        # conv layers following resizing
        self.resize_conv = ModuleList()
        for i in range(len(resize_channels)):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = resize_channels[i - 1]
            out_ch = resize_channels[i]
            self.resize_conv.append(
                nn.Sequential(
                    convNd(ndim, in_ch, out_ch, a=0.2),
                    nn.LeakyReLU(0.2),
                ))

        # final prediction layer
        self.conv = convNd(ndim, resize_channels[-1], ndim)

    @staticmethod
    def interpolate_(img, scale_factor=None, size=None, mode=None):
        """Wrapper for torch.nn.functional.interpolate."""
        if mode == 'nearest':
            mode = mode
        else:
            ndim = img.ndim - 2
            if ndim == 2:
                mode = 'bilinear'
            elif ndim == 3:
                mode = 'trilinear'
            else:
                raise ValueError(f'Data dimension ({ndim}) must be 2 or 3')

        y = nn.functional.interpolate(
            img,
            scale_factor=scale_factor,
            size=size,
            mode=mode,
        )
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # resize output of encoder-decoder
        x = self.interpolate_(x, size=self.output_size)

        # layers after resize
        for resize_layer in self.resize_conv:
            x = resize_layer(x)

        x = self.conv(x)

        return x


@MODELS.register_module()
class UpsampleFlow(nn.Upsample):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    Args:
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'trilinear'``
    """

    def __init__(self, mode='trilinear', **kwargs) -> None:
        super().__init__(mode=mode, **kwargs)
        self.linear_modes = {1: 'linear', 2: 'bilinear', 3: 'trilinear'}

    def forward_func(self, x: Tensor, **kwargs) -> Tensor:
        mode = self.mode if self.mode == 'nearest' \
            else self.linear_modes[x.ndim - 2]

        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    def forward(self, x: MultiTensor, **kwargs):
        if isinstance(x, Tensor):
            x = self.forward_func(x)
        elif isinstance(x, dict):
            for k in x.keys():
                x[k] = self.forward_func(x[k])
        elif is_seq_of(x, Tensor):
            x = [self.forward_func(i) for i in x]
        else:
            raise ValueError('`x` type is error')
        return x


@MODELS.register_module()
class IdentityFlow(BaseModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(init_cfg=None)

    def forward(self, x, **kwargs):
        return x


@MODELS.register_module()
class BiFlow(BaseModule):

    def __init__(
        self,
        in_channels,
        out_channels=3,
        kernel_size=3,
        bias=True,
        shared_conv=True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.shared_conv = shared_conv

        self.conv = build_conv_layer(
            cfg=dict(type=f'Conv{out_channels}d'),
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=1,
            bias=bias,
        )
        if not self.shared_conv:
            self.conv2 = build_conv_layer(
                cfg=dict(type=f'Conv{out_channels}d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
                groups=1,
                bias=bias,
            )

    def forward(self, x: Dict[str, torch.Tensor], training, **kwargs):
        if training:
            assert len(x) == 2, f'input len is not 2, got {len(x)}'
            flow_src = self.conv(x['source'])
            if self.shared_conv:
                flow_tgt = self.conv(x['target'])
            else:
                flow_tgt = self.conv2(x['target'])
            return dict(source=flow_src, target=flow_tgt)
        else:
            flow = self.conv(x)
            return flow

    def init_weights(self):
        normal_init(self.conv, mean=0, std=1e-5, bias=0)
        if not self.shared_conv:
            normal_init(self.conv2, mean=0, std=1e-5, bias=0)


def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.0):
    """
    Convolution of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation
    Returns:
        (BaseModule instance) Instance of convolution module of the specified dimension
    """
    conv_nd = getattr(nn, f'Conv{ndim}d')(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd
