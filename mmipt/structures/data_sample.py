# Copyright (c) MMIPT. All rights reserved.
from collections import abc
from copy import deepcopy
from itertools import chain
from typing import Any, Sequence

import numpy as np
import torch
from mmengine.structures import BaseDataElement

from mmipt.utils import all_to_tensor


class DataSample(BaseDataElement):
    """A data structure interface of MMipt. They are used as interfaces between
    different components, e.g., model, visualizer, evaluator, etc. Typically,
    DataSample contains all the information and data from ground-truth and
    predictions.

    `DataSample` inherits from `BaseDataElement`. See more details in:
      https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html
      Specifically, an instance of BaseDataElement consists of two components,
      - ``metainfo``, which contains some meta information.
      - ``data``, which contains the data used in the loop.
    """

    # source_key_in_results: target_key_in_metainfo
    META_KEYS = {
        'num_classes': 'num_classes',
        # For registration tasks
        'interp': 'interp',
        'source_shape': 'source_shape',
        'target_shape': 'target_shape',
        'source_path': 'source_img_path',
        'target_path': 'target_img_path',
        'source_img_path': 'source_img_path',
        'target_img_path': 'target_img_path',
        'source_seg_path': 'source_seg_path',
        'target_seg_path': 'target_seg_path',
        'source_mask_path': 'source_mask_path',
        'target_mask_path': 'target_mask_path',
        # For segmentation tasks
        'img_shape': 'img_shape',
        'seg_shape': 'seg_shape',
        'img_path': 'img_path',
        'seg_path': 'seg_path',
    }

    # source_key_in_results: target_key_in_datafield
    DATA_KEYS = {
        # For registration tasks
        'source_seg': 'source_seg',
        'target_seg': 'target_seg',
        'source_mask': 'source_mask',
        'target_mask': 'target_mask',
        # For segmentation tasks
        'seg': 'seg',
    }

    def set_predefined_data(self, data: dict) -> None:
        """set or change pre-defined key-value pairs in ``data_field`` by
        parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """

        metainfo = {
            self.META_KEYS[k]: v
            for (k, v) in data.items() if k in self.META_KEYS
        }
        self.set_metainfo(metainfo)

        data = {
            self.DATA_KEYS[k]: v
            for (k, v) in data.items() if k in self.DATA_KEYS
        }
        self.set_tensor_data(data)

    def set_tensor_data(self, data: dict) -> None:
        """convert input data to tensor, and then set or change key-value pairs
        in ``data_field`` by parameter ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            self.set_field(all_to_tensor(v), k, dtype=torch.Tensor)

    @classmethod
    def stack(cls, data_samples: Sequence['DataSample']) -> 'DataSample':
        """Stack a list of data samples to one. All tensor fields will be
        stacked at first dimension. Otherwise the values will be saved in a
        list.

        Args:
            data_samples (Sequence['DataSample']): A sequence of
                `DataSample` to stack.

        Returns:
            DataSample: The stacked data sample.
        """
        # 1. check key consistency
        keys = data_samples[0].keys()
        assert all([data.keys() == keys for data in data_samples])

        meta_keys = data_samples[0].metainfo_keys()
        assert all(
            [data.metainfo_keys() == meta_keys for data in data_samples])

        # 2. stack data
        stacked_data_sample = DataSample()
        for k in keys:
            values = [getattr(data, k) for data in data_samples]
            # 3. check type consistent
            value_type = type(values[0])
            assert all([type(val) == value_type for val in values])

            # 4. stack
            if isinstance(values[0], torch.Tensor):
                stacked_value = torch.stack(values)
            else:
                stacked_value = values
            stacked_data_sample.set_field(stacked_value, k)

        # 5. stack metainfo
        for k in meta_keys:
            values = [data.metainfo[k] for data in data_samples]
            stacked_data_sample.set_metainfo({k: values})

        return stacked_data_sample

    def split(self,
              allow_nonseq_value: bool = False) -> Sequence['DataSample']:
        """Split a sequence of data sample in the first dimension.

        Args:
            allow_nonseq_value (bool): Whether allow non-sequential data in
                split operation. If True, non-sequential data will be copied
                for all split data samples. Otherwise, an error will be
                raised. Defaults to False.

        Returns:
            Sequence[DataSample]: The list of data samples after splitting.
        """
        # 1. split
        data_sample_list = [DataSample() for _ in range(len(self))]
        for k in self.all_keys():
            stacked_value = self.get(k)
            if isinstance(stacked_value, torch.Tensor):
                # split tensor shape like (N, *shape) to N (*shape) tensors
                values = [v for v in stacked_value]
            elif isinstance(stacked_value, DataSample):
                values = stacked_value.split()
            else:
                if is_splitable_var(stacked_value):
                    values = stacked_value
                elif allow_nonseq_value:
                    values = [deepcopy(stacked_value)] * len(self)
                else:
                    raise TypeError(
                        f'\'{k}\' is non-sequential data and '
                        '\'allow_nonseq_value\' is False. Please check your '
                        'data sample or set \'allow_nonseq_value\' as True '
                        f'to copy field \'{k}\' for all split data sample.')

            field = 'metainfo' if k in self.metainfo_keys() else 'data'
            for data, v in zip(data_sample_list, values):
                data.set_field(v, k, field_type=field)

        return data_sample_list

    def __len__(self):
        """Get the length of the data sample."""

        value_length = []
        for v in chain(self.values(), self.metainfo_values()):
            if is_splitable_var(v):
                value_length.append(len(v))
            else:
                continue

        # NOTE: If length of values are not same or the current data sample
        # is empty, return length as 1
        if len(list(set(value_length))) != 1:
            return 1

        length = value_length[0]
        return length

    def get_value(self, key: str) -> Any:
        value = self.get(key)
        if isinstance(value, abc.Sequence):
            assert len(set(value)) == 1
            return value[0]
        else:
            return value

    def to_onehot(self, key: str) -> torch.Tensor:
        num_classes = self.get_value('num_classes')
        label = self.get(key, default=None)
        assert label.max().item() < num_classes
        onehot = torch.nn.functional.one_hot(
            label.squeeze().long(), num_classes=num_classes)
        return onehot


def is_splitable_var(var: Any) -> bool:
    """Check whether input is a splitable variable.

    Args:
        var (Any): The input variable to check.

    Returns:
        bool: Whether input variable is a splitable variable.
    """
    if isinstance(var, DataSample):
        return True
    if isinstance(var, torch.Tensor):
        return True
    if isinstance(var, np.ndarray):
        return True
    if isinstance(var, abc.Sequence) and not isinstance(var, str):
        return True
    return False
