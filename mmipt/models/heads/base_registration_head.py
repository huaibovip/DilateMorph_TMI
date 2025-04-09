# Copyright (c) MMIPT. All rights reserved.
from abc import abstractmethod
from typing import Optional, Sequence, Union

import torch
from mmengine.model import BaseModule
from mmengine.structures import BaseDataElement
from torch import Tensor
from torch.nn import functional as F

from mmipt.models.utils import Warp, flow_denorm, generate_grid
from mmipt.structures.data_sample import DataSample
from mmipt.utils import SampleInputs, TensorDict, to_channel_first


class BaseRegistrationHead(BaseModule):
    """Base class for registration.

    Args:
        img_size (Sequence[int]): size of input image.
        normalization (bool, optional): Normalize flow field. Default: False.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(self,
                 img_size: Sequence[int],
                 normalization: bool = False,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.img_size = img_size
        self.normalization = normalization

        # build warp layer
        self.warp = Warp(
            img_size,
            normalization=normalization,
            align_corners=True,
            padding_mode="zeros")

    def forward(
        self,
        flow: Tensor,
        input: Tensor,
        interp_mode: str,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            vec_flow (Tensor): flow field predicted by network.
            input (Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
        """
        return self.warp(flow, input, interp_mode=interp_mode)

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Placeholder of forward function when model training."""
        pass

    def forward_test(
        self,
        flow: Tensor,
        inputs: TensorDict,
        data_samples: SampleInputs = None,
        return_all: bool = True,
        **kwargs,
    ) -> BaseDataElement:
        """Forward function when model testing.

        Args:
            flow (Tensor): flow field predicted by network.
            inputs (Dict[str, Tensor]): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (BaseDataElement, List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            return_grid (bool): Whether to return the grid. Default: True.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        interp = data_samples.get_value('interp')
        num_classes = data_samples.get_value('num_classes')

        # warp segmentation with displacement field
        warped_seg = self.warp_seg(flow, data_samples.source_seg, num_classes,
                                   interp)

        predictions = DataSample(metainfo=data_samples.metainfo)
        predictions.set_tensor_data(dict(pred_seg=warped_seg))
        predictions.set_tensor_data(dict(target_seg=data_samples.target_seg))
        predictions.set_tensor_data(
            dict(pred_flow=flow_denorm(flow, self.normalization)))

        if return_all:
            # warp moving & grid with displacement field
            predictions.set_tensor_data(dict(pred_grid=self.warp_grid(flow)))
            predictions.set_tensor_data(
                dict(pred_img=self.warp_img(flow, inputs['source_img'])))
            predictions.set_tensor_data(dict(target_img=inputs['target_img']))

        return predictions

    @abstractmethod
    def losses(self, *args, **kwargs):
        """Placeholder for model computing losses."""
        pass

    def warp_grid(self, flow: Tensor, grid_step: int = 8) -> Tensor:
        grid_img = generate_grid(self.img_size, grid_step=grid_step).to(flow)
        grid_img = grid_img.repeat([flow.shape[0]] + [1] * (flow.ndim - 1))
        return self.warp(flow, grid_img, interp_mode='bilinear')

    def warp_img(self, flow: Tensor, img: Tensor) -> Tensor:
        return self.warp(flow, img, interp_mode='bilinear')

    def warp_seg(
        self,
        flow: Tensor,
        seg: Tensor,
        num_classes: int,
        mode: str = 'bilinear',
    ) -> Tensor:

        if mode == 'bilinear':
            seg_oh = to_channel_first(
                F.one_hot(seg.squeeze(1).long(), num_classes=num_classes))
            seg_oh = self.warp(flow, seg_oh.float(), interp_mode='bilinear')
            return torch.argmax(seg_oh, dim=1, keepdim=True)

        elif mode == 'nearest':
            return self.warp(flow, seg.float(), interp_mode='nearest').long()

        else:
            raise ValueError(f'not support interpolation {mode}')
