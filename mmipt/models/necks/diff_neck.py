# Copyright (c) MMIPT. All rights reserved.
from mmengine.model import BaseModule
from mmengine.utils import is_seq_of
from torch import Tensor

from mmipt.models.utils import Warp
from mmipt.registry import MODELS
from mmipt.utils import MultiTensor


@MODELS.register_module()
class DiffeomorphicTransform(BaseModule):
    """
    Integrates a flow field via scaling and squaring.
    """

    def __init__(
        self,
        img_size,
        nsteps=7,
        # scale=1.0,
        normalization: bool = False,
        padding_mode: str = 'zeros',
        init_cfg=None,
    ):
        super(DiffeomorphicTransform, self).__init__(init_cfg=init_cfg)
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        # self.scale = scale / (2**self.nsteps)

        self.normalization = normalization
        self.warp = Warp(
            img_size,
            normalization=normalization,
            align_corners=True,
            padding_mode=padding_mode,
        )

    def scale_square(self, flow: Tensor) -> Tensor:
        flow = flow * self.scale
        for _ in range(self.nsteps):
            flow = flow + self.warp(flow, flow)
        return flow

    def forward(self, flow: MultiTensor, *args, **kwargs):
        if isinstance(flow, Tensor):
            flow = self.scale_square(flow)
        elif isinstance(flow, dict):
            for k in flow.keys():
                flow[k] = self.scale_square(flow[k])
        elif is_seq_of(flow, Tensor):
            flow = [self.scale_square(i) for i in flow]
        else:
            raise ValueError('`flow` type is error')
        return flow
