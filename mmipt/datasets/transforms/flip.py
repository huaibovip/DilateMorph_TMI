# Copyright (c) MMIPT. All rights reserved.
from typing import List, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmipt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Randomly mirror flip across x, y (and z) for all keys.

    Args:
        keys (list[str]): A list specifying the keys whose values are
            modified.
        axes (tuple[int]):
    """

    def __init__(self, keys: List[str], axes: Sequence[int]):
        self.keys = keys
        self.axes = axes  # (1, 2, 3) for 3d and (1, 2) for 2d

    def get_params(self):
        params = [np.random.choice([True, False]) for _ in self.axes]

        return params

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        params = self.get_params()
        for key in self.keys:
            img = results[key]
            for axis_i, axis in enumerate(self.axes):
                if params[axis_i]:
                    img = np.flip(img, axis=axis)
                    img = np.ascontiguousarray(img)  # TODO
            results[key] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, '
                     f'axes={self.axes}')

        return repr_str
