# Copyright (c) MMIPT. All rights reserved.
from abc import ABCMeta
from typing import Optional

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmipt.models.heads import BaseRegistrationHead
from mmipt.models.utils import Compose
from mmipt.registry import MODELS
from mmipt.utils import (ConfigType, ForwardResults, MultiConfig, MultiTensor,
                         OptConfigType, SampleInputs, SampleList, TensorDict)


@MODELS.register_module()
class BaseRegister(BaseModel, metaclass=ABCMeta):
    """Base deformable model for image and volume registration.

    It must contain a generator that takes frames as inputs and outputs an
    interpolated frame. It also has a pixel-wise loss for training.

    Args:
        backbone (dict): Config for the backbone structure.
        flow (dict, Sequence[dict]): Config for the flow structure.
        head (dict): Config for the head structure.
        affine_model (dict): Config for the affine model structure.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.

    Attributes:
        init_cfg (dict, optional): Initialization config dict.
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`. Default: None.
    """

    def __init__(
            self,
            backbone: ConfigType,
            flow: MultiConfig,
            head: ConfigType,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = dict(
                type="RegDataPreprocessor"),
    ):
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # build backbone
        self.backbone = MODELS.build(backbone)

        # build flow neck
        if isinstance(flow, dict):
            self.flow = MODELS.build(flow)
        else:
            self.flow = Compose(flow)

        # build registration head
        self.head: BaseRegistrationHead
        self.head = MODELS.build(head)

    def convert_to_datasample(
        self,
        predictions: BaseDataElement,
        data_samples: BaseDataElement,
        inputs: Optional[torch.Tensor],
    ) -> SampleList:
        """Add predictions and destructed inputs (if passed) to data samples.

        Args:
            predictions (BaseDataElement): The predictions of the model.
            data_samples (BaseDataElement): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[BaseDataElement]: Modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, "img")
            data_samples.set_tensor_data({"input": destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

    def extract_feats(self, inputs: dict, **kwargs) -> MultiTensor:
        """Extract features from images.

        Args:
            imgs (dict): The concatenated input images.

        Returns:
            Tensor: The feature pyramid of the first input image
                and the feature pyramid of secode input image.
        """

        # extract features
        src, tgt = inputs["source_img"], inputs["target_img"]
        feats = self.backbone(src, tgt, **kwargs)
        flow = self.flow(feats, **kwargs)
        return flow

    def forward_train(self,
                      inputs: dict,
                      data_samples: SampleInputs = None,
                      **kwargs) -> TensorDict:
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (dict): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        flow = self.extract_feats(inputs, **kwargs)

        return self.head.forward_train(
            flow=flow, inputs=inputs, data_samples=data_samples, **kwargs)

    def forward_test(self,
                     inputs: dict,
                     data_samples: SampleInputs = None,
                     **kwargs) -> BaseDataElement:
        """Forward inference. Returns predictions of validation, testing, and
        simple inference.

        Args:
            inputs (dict): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            BaseDataElement: predictions.
        """

        flow = self.extract_feats(inputs, **kwargs)

        return self.head.forward_test(
            flow=flow, inputs=inputs, data_samples=data_samples, **kwargs)

    def forward(self,
                inputs: dict,
                data_samples: SampleInputs = None,
                mode: str = "tensor") -> ForwardResults:
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        ``forward`` method of BaseModel is an abstract method, its subclasses
        must implement this method.

        Accepts ``inputs`` and ``data_samples`` processed by
        :attr:`data_preprocessor`, and returns results according to mode
        arguments.

        During non-distributed training, validation, and testing process,
        ``forward`` will be called by ``BaseModel.train_step``,
        ``BaseModel.val_step`` and ``BaseModel.val_step`` directly.

        During distributed data parallel training process,
        ``MMSeparateDistributedDataParallel.train_step`` will first call
        ``DistributedDataParallel.forward`` to enable automatic
        gradient synchronization, and then call ``forward`` to get training
        loss.

        Args:
            inputs (dict): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``. Default: 'tensor'.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults:

                - If ``mode == loss``, return a ``dict`` of loss tensor used
                  for backward and logging.
                - If ``mode == predict``, return a ``list`` of
                  :obj:`BaseDataElement` for computing metric
                  and getting inference result.
                - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                  or ``dict`` or tensor for custom use.
        """
        if mode == "loss":
            return self.forward_train(inputs, data_samples, training=True)

        elif mode == "predict":
            predictions = self.forward_test(
                inputs, data_samples, training=False)
            # predictions = self.convert_to_datasample(predictions, data_samples, inputs)
            predictions = predictions.split()
            return predictions

        elif mode == "tensor":
            # return self.forward_tensor(inputs, data_samples)
            flow = self.extract_feats(inputs, training=False)
            source_img = inputs['source_img']
            # source_seg = data_samples.source_seg
            warped_img = self.head(
                flow, input=source_img, interp_mode='bilinear')
            return dict(flow=flow, warped_img=warped_img)
