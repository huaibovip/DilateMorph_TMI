# Copyright (c) MMIPT. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import autocast

from mmipt.registry import MODELS

_reduction_modes = ['none', 'mean', 'sum']


@MODELS.register_module()
class MILoss(nn.Module):
    """ Mutual Information loss

        Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self,
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 num_bins: int = 32,
                 sigma_ratio: float = 1.0,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.max_clip = maxval
        self.loss_weight = loss_weight

        bin_centers = np.linspace(minval, maxval, num=num_bins)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * sigma**2)

        vbc = torch.linspace(minval, maxval, num_bins, requires_grad=False)
        self.register_buffer('vol_bin_centers', vbc, persistent=False)

    def mutual_information(self, predict: Tensor, target: Tensor):
        predict = torch.clamp(predict, 0., self.max_clip)
        target = torch.clamp(target, 0, self.max_clip)

        target = target.view(target.shape[0], -1).unsqueeze(2)
        predict = predict.view(predict.shape[0], -1).unsqueeze(2)
        nb_voxels = predict.shape[1]  # total num of voxels

        # reshape bin centers
        shape = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = self.vol_bin_centers.reshape(shape).to(target)

        # compute image terms by approx. Gaussian dist.
        I_a = torch.exp(-self.preterm * torch.square(target - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(-self.preterm * torch.square(predict - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(
            torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    @autocast(enabled=False)
    def forward(self,
                predict: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            predict (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return -self.mutual_information(predict, target)


@MODELS.register_module()
class LocalMILoss(nn.Module):
    """
    Local Mutual Information loss for non-overlapping patches
    """

    def __init__(self,
                 patch_size: int = 5,
                 minval: float = 0.0,
                 maxval: float = 1.0,
                 num_bins: int = 32,
                 sigma_ratio: float = 1.0,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.max_clip = maxval
        self.num_bins = num_bins
        self.patch_size = patch_size
        self.loss_weight = loss_weight

        bin_centers = np.linspace(minval, maxval, num=num_bins)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = 1 / (2 * sigma**2)

        vbc = torch.linspace(minval, maxval, num_bins, requires_grad=False)
        self.register_buffer('vol_bin_centers', vbc, persistent=False)

    def local_mutual_information(self, predict: Tensor, target: Tensor):
        predict = torch.clamp(predict, 0., self.max_clip)
        target = torch.clamp(target, 0, self.max_clip)

        # reshape bin centers
        shape = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = self.vol_bin_centers.reshape(shape).to(target)

        # making image paddings
        if len(list(predict.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(predict.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2,
                       x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(predict.size())[2:]) == 2:
            ndim = 2
            x, y = list(predict.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0,
                       0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(
                list(predict.size())))
        target = F.pad(target, padding, "constant", 0)
        predict = F.pad(predict, padding, "constant", 0)

        # reshaping images into non-overlapping patches
        if ndim == 3:
            target_patch = torch.reshape(
                target, (target.shape[0], target.shape[1],
                         (x + x_r) // self.patch_size, self.patch_size,
                         (y + y_r) // self.patch_size, self.patch_size,
                         (z + z_r) // self.patch_size, self.patch_size))
            target_patch = target_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            target_patch = target_patch.reshape(-1, self.patch_size**3, 1)

            predict_patch = torch.reshape(
                predict, (predict.shape[0], predict.shape[1],
                          (x + x_r) // self.patch_size, self.patch_size,
                          (y + y_r) // self.patch_size, self.patch_size,
                          (z + z_r) // self.patch_size, self.patch_size))
            predict_patch = predict_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            predict_patch = predict_patch.reshape(-1, self.patch_size**3, 1)
        else:
            target_patch = torch.reshape(
                target, (target.shape[0], target.shape[1],
                         (x + x_r) // self.patch_size, self.patch_size,
                         (y + y_r) // self.patch_size, self.patch_size))
            target_patch = target_patch.permute(0, 1, 2, 4, 3, 5)
            target_patch = target_patch.reshape(-1, self.patch_size**2, 1)

            predict_patch = torch.reshape(
                predict, (predict.shape[0], predict.shape[1],
                          (x + x_r) // self.patch_size, self.patch_size,
                          (y + y_r) // self.patch_size, self.patch_size))
            predict_patch = predict_patch.permute(0, 1, 2, 4, 3, 5)
            predict_patch = predict_patch.reshape(-1, self.patch_size**2, 1)

        # compute MI
        I_a_patch = torch.exp(-self.preterm * torch.square(target_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(-self.preterm *
                              torch.square(predict_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size**ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(
            torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self,
                predict: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return -self.local_mutual_information(predict, target)
