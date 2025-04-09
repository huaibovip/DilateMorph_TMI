# Copyright (c) MMIPT. All rights reserved.
from logging import WARNING
from typing import List, Optional, Union

import torch
from mmengine import print_log
from mmengine.model import BaseDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor

from mmipt.registry import MODELS
from mmipt.structures import DataSample
from mmipt.utils.typing import SampleList

CastData = Union[tuple, dict, list, Tensor, DataSample]


@MODELS.register_module()
class RegDataPreprocessor(BaseDataPreprocessor):
    """Image / Volume pre-processor for registration tasks.

    See base class ``BaseDataPreprocessor`` for detailed information.
    Workflow as follow :
    - Collate and move data to the target device.
    - Stack inputs to batch_inputs.
    - Conversion
    - Normalize volume with defined std and mean.

    Args:
        mean (Union[float, int], float or int, optional): The pixel mean
            of volume channels. Noted that normalization operation is performed
            *after data conversion*. If it is not specified, volumes
            will not be normalized. Defaults None.
        std (Union[float, int], float or int, optional): The pixel
            standard deviation of volume channels. Noted that normalization
            operation is performed *after data conversion*. If it is
            not specified, volumes will not be normalized. Defaults None.
        data_keys (List[str] or str): Keys to preprocess in data samples.
            Defaults is None.
        stack_data_sample (bool): Whether stack a list of data samples to one
            data sample. Only support with input data samples are
            `DataSamples`. Defaults to True.
    """

    def __init__(self,
                 norm_type: Optional[str] = None,
                 mean: Union[float, int] = None,
                 std: Union[float, int] = None,
                 data_keys: Union[List[str], str] = None,
                 stack_data_sample: bool = True):
        super().__init__(non_blocking=False)
        assert norm_type in [
            'max-min', 'mean-std', 'zcore', None
        ], ('expect norm "type" be "max-min", "mean-std" or "zcore", but get {}'
            .format(norm_type))

        if norm_type == 'mean-std':
            assert (mean is None) == (std is None), (
                'mean and std should be both None or float')
            self.register_buffer('mean', torch.tensor(mean), False)
            self.register_buffer('std', torch.tensor(std), False)
        self.norm_type = norm_type

        if data_keys is not None and not isinstance(data_keys, list):
            self.data_keys = [data_keys]
        else:
            self.data_keys = data_keys

        self.stack_data_sample = stack_data_sample

    def _do_conversion(self, inputs: Tensor, ndim: int = 3) -> Tensor:
        """Conduct channel order conversion for *a batch of inputs*, and return
        the converted inputs and order after conversion.

        Args:
            inputs (Tensor): Tensor with shape (B, C, [D], H, W).
        """
        return inputs

    def _do_normalization(self, inputs: Tensor, ndim: int = 3) -> Tensor:
        """
        Args:
            inputs (Tensor): Tensor with shape (B, C, [D], H, W).
        """
        if self.norm_type == 'mean-std':
            shape = [-1] + [1] * ndim
            mean, std = self.mean.view(shape), self.std.view(shape)
            inputs = (inputs.float() - mean) / std
        elif self.norm_type == 'max-min':
            vmin = inputs.min()
            inputs = (inputs.float() - vmin) / (inputs.max() - vmin)
        elif self.norm_type == 'zscore':
            shape = [-1] + [1] * ndim
            mean, std = inputs.mean().view(shape), inputs.std().view(shape)
            inputs = (inputs.float() - mean) / std

        return inputs

    def __prep_tensor(self, inputs: Tensor) -> Tensor:
        """Preprocess a batch of tensor.

        Args:
            inputs (Tensor): Tensor with shape (N, C, H, W),
                or (N, C, D, H, W) to preprocess.

        Returns:
            Tensor: The preprocessed tensor.
        """

        assert inputs.ndim in [4, 5], ('The input should be a (N, C, H, W) '
                                       'or (N, C, D, H, W) tensor, but got a '
                                       f'tensor with shape: {inputs.shape}')

        ndim = len(inputs.shape[2:])
        inputs = self._do_conversion(inputs, ndim)
        inputs = self._do_normalization(inputs, ndim)
        return inputs

    def _prep_tensor_list(self, inputs: List[Tensor]) -> Tensor:
        """Preprocess a list of tensor.

        Args:
            inputs (List[Tensor]): Tensor list to be preprocess.
            The size is [channels, [depth], height, width]

        Returns:
            Tensor: The preprocessed tensor.
        """
        inputs = torch.stack(inputs, dim=0)
        # preprocess tensor
        ndim = len(inputs.shape[2:])
        inputs = self._do_conversion(inputs, ndim)
        inputs = self._do_normalization(inputs, ndim)
        return inputs

    def _prep_tensor_dict(self, batch_inputs: dict) -> dict:
        """Preprocess a dict of inputs.

        Args:
            inputs (dict): Input dict.

        Returns:
            Tuple[dict, List[DataSample]]: The preprocessed dict.
        """
        for k, inputs in batch_inputs.items():
            if isinstance(inputs, list):
                assert all([
                    isinstance(inp, Tensor) for inp in inputs
                ]), 'Only support stack list of Tensor in inputs dict.'
                batch_inputs[k] = self._prep_tensor_list(inputs)
            elif isinstance(inputs, Tensor):
                batch_inputs[k] = self.__prep_tensor(inputs)
        return batch_inputs

    def _prep_data_sample(self, data_samples: SampleList,
                          training: bool) -> Union[DataSample, SampleList]:
        """Preprocess data samples.

        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """
        if data_samples is None:
            return None

        for data_sample in data_samples:
            if not self.data_keys:
                break
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    print_log(f'Cannot find key \'{key}\' in data sample.',
                              'current', WARNING)
                    break
                # TODO do something

        if self.stack_data_sample:
            return DataSample.stack(data_samples)

        return data_samples

    def forward(self, data: dict, training: bool = False) -> dict:
        """Pre-process data as configured based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Input data to process.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Dict: Data in the same format as the model input.
        """

        # collates and moves data to the target device.
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        # process inputs
        if isinstance(inputs, Tensor):
            inputs = self.__prep_tensor(inputs)
        elif is_seq_of(inputs, Tensor):
            inputs = self._prep_tensor_list(inputs)
        elif isinstance(inputs, dict):
            inputs = self._prep_tensor_dict(inputs)
        elif is_seq_of(inputs, dict):
            # convert list of dict to dict of list
            keys = inputs[0].keys()
            dict_input = {k: [inp[k] for inp in inputs] for k in keys}
            inputs = self._prep_tensor_dict(dict_input)
        else:
            raise ValueError('Only support following inputs types: '
                             '\'Tensor\', \'List[Tensor]\', \'dict\', '
                             '\'List[dict]\'. But receive '
                             f'\'{type(inputs)}\'.')

        # process data samples
        data_samples = self._prep_data_sample(data_samples, training)

        return dict(inputs=inputs, data_samples=data_samples)
