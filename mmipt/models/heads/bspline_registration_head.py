# Copyright (c) MMIPT. All rights reserved.
from mmengine.structures import BaseDataElement
from torch import Tensor
from torch.nn import functional as F

from mmipt.registry import MODELS
from mmipt.utils import SampleInputs, TensorDict, TensorList, to_channel_first
from .deformable_registration_head import DeformableRegistrationHead


@MODELS.register_module()
class BSplineRegistrationHead(DeformableRegistrationHead):
    """Head for bspline registration.

    Args:
        img_size (Sequence[int]): size of input image.
        loss_sim (dict): Config for image similarity loss.
        loss_reg (dict): Config for deformation field regularization loss.
        loss_seg (dict): Config for segmentation loss. Default: None.
        normalization (bool, optional): Normalize flow field. Default: False.
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(BSplineRegistrationHead, self).__init__(*args, **kwargs)

    def forward(
        self,
        flow: TensorList,
        input: Tensor,
        interp_mode: str,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            flow (Sequence[Tensor]): velocity and displacement
                fields predicted by network.
            input (Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            interp_mode (str): interpolation mode. ["nearest", "bilinear", "bicubic"]
        """
        _, disp = flow
        return self.warp(disp, input, interp_mode=interp_mode)

    def forward_train(
        self,
        flow: TensorList,
        inputs: TensorDict,
        data_samples: SampleInputs = None,
        **kwargs,
    ) -> TensorDict:
        """Forward function when model training.

        Args:
            flow (Sequence[Tensor]): velocity and displacement
                fields predicted by network.
            inputs (Dict[Tensor]): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (BaseDataElement, List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Dict[str, Tensor]: The dict of losses.
        """
        source_img = inputs['source_img']
        target_img = inputs['target_img']
        velocity, disp = flow

        # warp image with displacement field
        warped_img = self.warp(disp, source_img, interp_mode='bilinear')

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
                disp, source_seg_oh.float(), interp_mode='nearest')

            return self.losses(
                velocity,
                target_img,
                warped_img,
                target_seg_oh,
                warped_seg_oh,
            )

        return self.losses(velocity, target_img, warped_img)

    def forward_test(
        self,
        flow: TensorList,
        inputs: Tensor,
        data_samples: SampleInputs = None,
        **kwargs,
    ) -> BaseDataElement:
        """Forward function when model testing.

        Args:
            flow (Sequence[Tensor]): velocity and displacement
                fields predicted by network.
            inputs (Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (BaseDataElement, List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        return super().forward_test(
            flow[1],
            inputs=inputs,
            data_samples=data_samples,
            **kwargs)
