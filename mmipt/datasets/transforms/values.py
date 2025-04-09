# Copyright (c) MMIPT. All rights reserved.
from copy import deepcopy

from mmcv.transforms import BaseTransform

from mmipt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CopyValues(BaseTransform):
    """Copy the value of source keys to destination keys.

    It does the following: results[dst_key] = results[src_key] for
    (dst_key, src_key) in dict(dst_key=src_key).

    Args:
        meta (dict): dict(dst_key=src_key).
    """

    def __init__(self, meta: dict) -> None:

        if not isinstance(meta, dict):
            raise AssertionError('"meta" must be dict.')

        self.meta = meta

    def transform(self, results: dict) -> dict:
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """

        for (dst_key, src_key) in self.meta.items():
            results[dst_key] = deepcopy(results[src_key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(meta={self.meta})')

        return repr_str


@TRANSFORMS.register_module()
class SetValues(BaseTransform):
    """Set value to destination keys.

    It does the following: results[key] = value

    Added keys are the keys in the meta.

    Args:
        meta (dict): The meta to update.
    """

    def __init__(self, meta: dict) -> None:

        if not isinstance(meta, dict):
            raise AssertionError('"meta" must be dict.')

        self.meta = meta

    def transform(self, results: dict) -> dict:
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """
        if self.meta is None or len(self.meta) == 0:
            return results

        results.update(deepcopy(self.meta))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(meta={self.meta})')

        return repr_str
