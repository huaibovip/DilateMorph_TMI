# Copyright (c) MMIPT. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmipt.registry import MODELS

_reduction_modes = ['none', 'mean', 'sum']


@MODELS.register_module()
class NCCLoss(nn.Module):
    """Local (over window) normalized cross correlation loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self,
                 window: int = 9,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.window = window
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self,
                predict: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        Ii = target  # /100
        Ji = predict  # /100
        device = target.device

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.shape) - 2
        assert ndims in [
            1, 2, 3
        ], ("volumes should be 1 to 3 dimensions. found: %d" % ndims)

        # set window size
        win = [self.window] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device) / float(np.prod(win))
        sum_filt.requires_grad = False

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)
        padding = win[0] // 2

        # compute CC squares
        mu1 = conv_fn(Ii, sum_filt, padding=padding)
        mu2 = conv_fn(Ji, sum_filt, padding=padding)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, sum_filt, padding=padding) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, sum_filt, padding=padding) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, sum_filt, padding=padding) - mu1_mu2

        cc = (sigma12 * sigma12) / (sigma1_sq * sigma2_sq + 1e-5)
        loss = 1 - torch.mean(cc)
        return loss * self.loss_weight


@MODELS.register_module()
class NCCVxmLoss(nn.Module):
    """Local (over window) normalized cross correlation loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self,
                 window: int = 9,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.window = window
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self,
                predict: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        Ii = target
        Ji = predict
        device = target.device

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(Ii.shape) - 2
        assert ndims in [
            1, 2, 3
        ], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.window] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)
        sum_filt.requires_grad = False

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)
        padding = win[0] // 2

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        loss = 1.0 - torch.mean(cc)
        return loss * self.loss_weight
