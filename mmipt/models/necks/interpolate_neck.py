# Copyright (c) MMIPT. All rights reserved.
from typing import Optional
from mmengine.model import BaseModule
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmipt.registry import MODELS
from mmipt.utils import MultiTensor


@MODELS.register_module()
class ResizeTransform(BaseModule):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self,
                 scale_factor: float,
                 mode: Optional[str] = None,
                 align_corners: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def resize(self, x):
        if self.mode is None:
            ndim = len(x.shape) - 2
            assert ndim in [2, 3]
            self.mode = 'bilinear' if ndim == 2 else 'trilinear'

        if self.scale_factor < 1.0:
            # resize first to save memory
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)
            x = self.scale_factor * x

        elif self.scale_factor > 1.0:
            # multiply first to save memory
            x = self.scale_factor * x
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)
        # don't do anything if resize is 1
        return x

    def forward(self, flow: MultiTensor, *args, **kwargs):
        if isinstance(flow, Tensor):
            flow = self.resize(flow)
        elif isinstance(flow, dict):
            for k in flow.keys():
                flow[k] = self.resize(flow[k])
        elif is_seq_of(flow, Tensor):
            flow = [self.resize(i) for i in flow]
        else:
            raise ValueError('`flow` type is error')
        return flow
