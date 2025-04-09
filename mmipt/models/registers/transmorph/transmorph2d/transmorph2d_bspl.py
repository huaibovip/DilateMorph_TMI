# Copyright (c) MMIPT. All rights reserved.
from torch import nn as nn

from mmipt.registry import MODELS
from .transmorph2d import TransMorph2dHalf


@MODELS.register_module()
class TransMorph2dBSpline(TransMorph2dHalf):

    def __init__(
            self,
            window_size,
            patch_size=4,
            flow_dim=48,  # important
            in_dim=2,
            embed_dim=96,
            depths=(2, 2, 4, 2),
            num_heads=(4, 4, 8, 8),
            mlp_ratio=4,
            pat_merg_rf=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            out_indices=(0, 1, 2, 3),
    ):
        super(TransMorph2dBSpline, self).__init__(
            window_size=window_size,
            patch_size=patch_size,
            flow_dim=flow_dim,
            in_dim=in_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            pat_merg_rf=pat_merg_rf,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            out_indices=out_indices)
