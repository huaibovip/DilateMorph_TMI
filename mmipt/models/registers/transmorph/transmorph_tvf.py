# Copyright (c) MMIPT. All rights reserved.
import torch
import torch.nn as nn

from mmipt.models.necks import DefaultFlow
from mmipt.models.utils import Warp
from mmipt.registry import MODELS
from . import transmorph as TM


@MODELS.register_module()
class TransMorphTVFForward(nn.Module):
    """
    TransMorph-TVF model

    Chen, J., Frey, E. C., & Du, Y. (2022).
    Unsupervised Learning of Diffeomorphic Image Registration via TransMorph.
    In International Workshop on Biomedical Image Registration (pp. 96-102). Springer, Cham.
    """

    def __init__(
            self,
            img_size,
            window_size,
            patch_size=4,
            flow_dim=16,
            in_dim=2,
            out_dim=3,
            embed_dim=128,
            depths=(2, 2, 12, 2),
            num_heads=(4, 4, 8, 16),
            mlp_ratio=4,
            pat_merg_rf=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            out_indices=(0, 1, 2, 3),
            time_steps=12,
    ):
        super().__init__()
        self.time_steps = time_steps

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.transformer = TM.SwinTransformer(
            patch_size=patch_size,
            in_dim=in_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pat_merg_rf=pat_merg_rf,
            out_indices=out_indices)
        self.up0 = TM.DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4,
            use_batchnorm=False)
        self.up1 = TM.DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2,
            use_batchnorm=False)
        self.up2 = TM.DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim,
            use_batchnorm=False)

        self.stn_down = Warp([s // 2 for s in img_size])

        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for _ in range(self.time_steps):
            self.cs.append(
                TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.up3s.append(
                TM.DecoderBlock(
                    embed_dim,
                    flow_dim,
                    skip_channels=embed_dim // 2,
                    use_batchnorm=False))
            self.reg_heads.append(
                DefaultFlow(
                    in_channels=flow_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                ))

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)

        source_d = self.avg_pool(source)
        pool_x = self.avg_pool(x)

        feats = self.transformer(x)
        x = self.up0(feats[-1], feats[-2])
        x = self.up1(x, feats[-3])
        xx = self.up2(x, feats[-4])

        # flow integration
        def_src = pool_x[:, 0:1, ...]
        def_tar = pool_x[:, 1:2, ...]
        flow_previous = 0
        flows = []
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_src, def_tar), dim=1))
            x = self.up3s[t](xx, f_out)
            flow = self.reg_heads[t](x)
            flows.append(flow)
            flow_new = flow_previous + self.stn_down(flow, flow)
            def_src = self.stn_down(flow_new, source_d)
            flow_previous = flow_new
        return flow_new


@MODELS.register_module()
class TransMorphTVFBackward(nn.Module):

    def __init__(
            self,
            img_size,
            window_size,
            patch_size=4,
            flow_dim=16,
            in_dim=2,
            out_dim=3,
            embed_dim=128,
            depths=(2, 2, 12, 2),
            num_heads=(4, 4, 8, 16),
            mlp_ratio=4,
            pat_merg_rf=4,
            qkv_bias=False,
            drop_rate=0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            out_indices=(0, 1, 2, 3),
            time_steps=12,
    ):
        super(TransMorphTVFBackward, self).__init__()
        self.time_steps = time_steps

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.transformer = TM.SwinTransformer(
            patch_size=patch_size,
            in_dim=in_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            out_indices=out_indices,
            pat_merg_rf=pat_merg_rf)
        self.up0 = TM.DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4,
            use_batchnorm=False)
        self.up1 = TM.DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2,
            use_batchnorm=False)
        self.up2 = TM.DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim,
            use_batchnorm=False)

        self.stn_down = Warp([s // 2 for s in img_size])

        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for _ in range(self.time_steps):
            self.cs.append(
                TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False))
            self.up3s.append(
                TM.DecoderBlock(
                    embed_dim,
                    flow_dim,
                    skip_channels=embed_dim // 2,
                    use_batchnorm=False))
            self.reg_heads.append(
                DefaultFlow(
                    in_channels=flow_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                ))
        self.tri_up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)

        source_d = self.avg_pool(source)
        pool_x = self.avg_pool(x)

        feats = self.transformer(x)
        x = self.up0(feats[-1], feats[-2])
        x = self.up1(x, feats[-3])
        xx = self.up2(x, feats[-4])

        # flow integration
        def_src = pool_x[:, 0:1, ...]
        def_tar = pool_x[:, 1:2, ...]
        flow_previous = 0
        flow_inv_previous = 0
        flows = []
        flows_out = []
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_src, def_tar), dim=1))
            x = self.up3s[t](xx, f_out)
            flow = self.reg_heads[t](x)
            flows.append(flow)
            flow_new = flow_previous + self.stn_down(flow, flow)
            flows_out.append(flow_new)
            flow_inv = flow_inv_previous + self.stn_down(-flow, -flow)
            def_src = self.stn_down(flow_new, source_d)
            flow_previous = flow_new
            flow_inv_previous = flow_inv

        flow = self.tri_up(flow_new)
        flow_inv = self.tri_up(flow_inv)
        # out = self.stn(source, flow)
        # return out, flow, flow_inv, flows, flows_out
        return flow, flow_inv
