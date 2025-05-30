# Copyright (c) MMIPT. All rights reserved.
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks.drop import DropPath
from torch import nn
from torch.nn import functional as F

from mmipt.registry import MODELS
from . import transmorph as TM


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = drop

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, self.drop, training=True)
        x = self.fc2(x)
        x = F.dropout(x, self.drop, training=True)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias.

    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                (2 * window_size[2] - 1),
                num_heads,
            ))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        # 3, Wh, Ww, Wt
        vectors = [coords_h, coords_w, coords_t]
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            coords = torch.stack(torch.meshgrid(vectors, indexing='ij'))
        else:
            coords = torch.stack(torch.meshgrid(vectors))
        # 3, Wh*Ww*Wt
        coords_flatten = torch.flatten(coords, 1)

        # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= ((2 * self.window_size[1] - 1) *
                                     (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # Wh*Ww*Wt, Wh*Ww*Wt
        relative_pos_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_pos_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = drop

    def forward(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        # (num_windows*B, Wh*Ww*Wt, C)
        B_, N, C = x.shape
        shape = (B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = self.qkv(x).reshape(*shape).permute(2, 0, 3, 1, 4)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_pos_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] *
                self.window_size[2], self.window_size[0] *
                self.window_size[1] * self.window_size[2],
                -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
        # nH, Wh*Ww*Wt, Wh*Ww*Wt
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N) +
                mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = F.dropout(attn, self.drop, training=True)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = F.dropout(x, self.drop, training=True)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        mc_drop (float): Ratio for MC dropout layers. Default: 0.15
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mc_drop=0.15,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= min(self.shift_size) < min(self.window_size)
        ), 'shift_size must in 0-window_size, shift_sz: {}, win_size: {}'.format(
            self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=mc_drop,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mc_drop,
        )

        self.H = None
        self.W = None
        self.T = None

    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, 'input feature has wrong size'

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] -
                 H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] -
                 W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] -
                 T % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(
                x,
                shifts=(
                    -self.shift_size[0],
                    -self.shift_size[1],
                    -self.shift_size[2],
                ),
                dims=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # nW*B, window_size, window_size, window_size, C
        x_windows = TM.window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size*window_size, C
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(
            -1,
            self.window_size[0],
            self.window_size[1],
            self.window_size[2],
            C,
        )
        # B H' W' T' C
        shifted_x = TM.window_reverse(attn_windows, self.window_size, Hp, Wp,
                                      Tp)

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x,
                shifts=(
                    self.shift_size[0],
                    self.shift_size[1],
                    self.shift_size[2],
                ),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        pat_merg_rf=2,
        mc_drop=0.15,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (
            window_size[0] // 2,
            window_size[1] // 2,
            window_size[2] // 2,
        )
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2,
                    window_size[1] // 2,
                    window_size[2] // 2,
                ),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                mc_drop=mc_drop,
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        # 1 Hp Wp Tp 1
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size[0]),
            slice(-self.shift_size[0], None),
        )
        w_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size[1]),
            slice(-self.shift_size[1], None),
        )
        t_slices = (
            slice(0, -self.window_size[2]),
            slice(-self.window_size[2], -self.shift_size[2]),
            slice(-self.shift_size[2], None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        # nW, window_size, window_size, window_size, 1
        mask_windows = TM.window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1,
            self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0,
            float(-100.0),
        ).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class SwinTransformer(nn.Module):
    """Swin Transformer

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_dim (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        window_size,
        patch_size=4,
        in_dim=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        pat_merg_rf=2,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        mc_drop=0.15,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.mc_drop = mc_drop

        # split image into non-overlapping patches
        self.patch_embed = TM.PatchEmbed(
            patch_size=patch_size,
            in_dim=in_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, dpr, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=TM.PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pat_merg_rf=pat_merg_rf,
                mc_drop=mc_drop,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                shape = (-1, H, W, T, self.num_features[i])
                out = x_out.view(*shape).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()


@MODELS.register_module()
class TransMorphBayes(nn.Module):

    def __init__(
            self,
            window_size,
            patch_size=4,
            flow_dim=16,
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
            mc_drop=0.15,
    ):
        super().__init__()
        self.mc_drop = mc_drop

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = TM.Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = TM.Conv3dReLU(2, flow_dim, 3, 1, use_batchnorm=False)

        self.transformer = SwinTransformer(
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
            out_indices=out_indices,
            mc_drop=mc_drop,
        )

        self.up0 = TM.DecoderBlock(
            embed_dim * 8,
            embed_dim * 4,
            skip_channels=embed_dim * 4,
            use_batchnorm=False,
        )
        self.up1 = TM.DecoderBlock(
            embed_dim * 4,
            embed_dim * 2,
            skip_channels=embed_dim * 2,
            use_batchnorm=False,
        )  # 384, 20, 20, 64
        self.up2 = TM.DecoderBlock(
            embed_dim * 2,
            embed_dim,
            skip_channels=embed_dim,
            use_batchnorm=False,
        )  # 384, 40, 40, 64
        self.up3 = TM.DecoderBlock(
            embed_dim,
            embed_dim // 2,
            skip_channels=embed_dim // 2,
            use_batchnorm=False,
        )  # 384, 80, 80, 128
        self.up4 = TM.DecoderBlock(
            embed_dim // 2,
            flow_dim,
            skip_channels=flow_dim,
            use_batchnorm=False,
        )  # 384, 160, 160, 256

    def forward(self, source, target, **kwargs):
        x = torch.cat([source, target], dim=1)

        x_s0 = x.clone()
        x_s1 = self.avg_pool(x)
        f4 = self.c1(x_s1)
        f4 = F.dropout3d(f4, self.mc_drop, training=True)
        f5 = self.c2(x_s0)
        f5 = F.dropout3d(f5, self.mc_drop, training=True)

        feats = self.transformer(x)

        x = self.up0(feats[-1], feats[-2])
        x = self.up1(x, feats[-3])
        x = self.up2(x, feats[-4])
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        return x
