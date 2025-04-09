# Copyright (c) MMIPT. All rights reserved.
from typing import Optional, Sequence, Union

from torch import Tensor
from torch.nn import functional as F

from mmipt.registry import MODELS
from mmipt.utils import SampleInputs, TensorDict, to_channel_first
from .base_registration_head import BaseRegistrationHead


@MODELS.register_module()
class DeformableRegistrationHead(BaseRegistrationHead):
    """Head for deformable registration.

    Args:
        img_size (Sequence[int]): size of input image.
        loss_sim (dict): Config for image similarity loss.
        loss_reg (dict): Config for deformation field regularization loss.
        loss_seg (dict): Config for segmentation loss. Default: None.
        normalization (bool, optional): Normalize flow field. Default: False.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(
        self,
        img_size: Sequence[int],
        loss_sim: dict,
        loss_reg: dict,
        loss_seg: Optional[dict] = None,
        normalization: bool = False,
        init_cfg: Optional[Union[dict, list]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            img_size=img_size,
            normalization=normalization,
            init_cfg=init_cfg,
            **kwargs)

        self.with_seg_loss = loss_seg is not None

        # build losses
        self.loss_sim = MODELS.build(loss_sim)
        self.loss_reg = MODELS.build(loss_reg)
        if self.with_seg_loss:
            self.loss_seg = MODELS.build(loss_seg)

    def forward_train(
        self,
        flow: Tensor,
        inputs: TensorDict,
        data_samples: SampleInputs = None,
        **kwargs,
    ) -> TensorDict:
        """Forward function when model training.

        Args:
            flow (Tensor): flow field predicted by network.
            inputs (Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (BaseDataElement, List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        source_img = inputs['source_img']
        target_img = inputs['target_img']

        # warp image with displacement field
        warped_img = self.warp(flow, source_img, interp_mode='bilinear')

        if self.with_seg_loss:
            num_classes = data_samples.num_classes[0]
            source_seg_oh = to_channel_first(
                F.one_hot(
                    data_samples.source_seg.squeeze(1).long(),
                    num_classes=num_classes))
            target_seg_oh = to_channel_first(
                F.one_hot(
                    data_samples.target_seg.squeeze(1).long(),
                    num_classes=num_classes))

            # warp one-hot label with displacement field
            warped_seg_oh = self.warp(
                flow, source_seg_oh.float(), interp_mode='bilinear')

            return self.losses(
                flow,
                target_img,
                warped_img,
                target_seg_oh,
                warped_seg_oh,
            )

        return self.losses(flow, target_img, warped_img)

    def losses(
        self,
        flow: Tensor,
        target: Tensor,
        predict: Tensor,
        target_seg: Optional[Tensor] = None,
        predict_seg: Optional[Tensor] = None,
    ) -> TensorDict:
        """Compute optical flow loss.

        Args:
            flow (Tensor): flow field predicted by network.
            target (Tensor): The ground truth of optical flow.
            predict (Tensor): The ground truth of optical flow.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        losses = dict()
        losses['loss_reg'] = self.loss_reg(flow)
        losses['loss_sim'] = self.loss_sim(target, predict)

        if self.with_seg_loss:
            losses['loss_seg'] = self.loss_seg(target_seg, predict_seg)

        return losses

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_size={self.img_size}, '
                     f'loss_sim={self.loss_sim}, '
                     f'loss_reg={self.loss_reg}')
        if self.with_seg_loss:
            repr_str += f', loss_seg={self.loss_seg}'
        repr_str += ')'
        return repr_str
