# Copyright (c) MMIPT. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn

from mmipt.registry import MODELS

_reduction_modes = ['none', 'mean', 'sum']


@MODELS.register_module('GradientDiffusionLoss')
@MODELS.register_module()
class GradLoss(nn.Module):
    """N-D gradient loss.

    normalization (bool): Whether the field is normalized. Default: False.
    """

    def __init__(self,
                 penalty='l2',
                 loss_weight: float = 1.0,
                 normalization: bool = False,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.penalty = penalty
        self.loss_weight = loss_weight
        self.normalization = normalization
        self.reduction = reduction

        if self.penalty not in ['l1', 'l2']:
            raise ValueError(f'Unsupported penalty: {self.penalty}. '
                             f'Supported ones are: l1, l2')
        if self.reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction}. '
                             f'Supported ones are: {_reduction_modes}')

    def forward(self,
                flow: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            flow (Tensor): of shape (N, C, H, W). Predicted flow tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        ndims = len(flow.shape) - 2
        if ndims == 2:
            dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
            dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

            if self.normalization:
                height, width = flow.shape[2:]
                dy = dy / 2 * height
                dx = dx / 2 * width

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

        elif ndims == 3:
            dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
            dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
            dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

            if self.normalization:
                depth, height, width = flow.shape[2:]
                dy = dy / 2 * depth
                dx = dx / 2 * height
                dz = dz / 2 * width

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

        else:
            raise NotImplementedError(
                "volumes should be 1 to 3 dimensions. found: %d" % ndims)

        return grad * self.loss_weight
