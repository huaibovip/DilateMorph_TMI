# Copyright (c) MMIPT. All rights reserved.
from typing import Dict, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmipt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SegmentNormalize(BaseTransform):

    def __init__(self, keys, seg_table: Sequence[int]):
        self.keys = keys
        # backgroud label is 0
        self.seg_table = np.array(seg_table)

    def transform(self, results: Dict):
        for key in self.keys:
            seg = results.get(key, None)
            if seg is None:
                raise ValueError(f'{key} is None.')

            seg_out = np.zeros_like(seg)
            for i in range(len(self.seg_table)):
                seg_out[seg == self.seg_table[i]] = i + 1
            results[key] = seg_out
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, '
                    f'seg_table={self.seg_table})')

        return repr_str


@TRANSFORMS.register_module()
class IXISegmentNormalize(SegmentNormalize):

    def __init__(self, keys):
        # backgroud label is 0
        seg_table = [
            2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 31, 41,
            42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63
        ]
        super().__init__(keys=keys, seg_table=seg_table)


@TRANSFORMS.register_module()
class LPBASegmentNormalize(SegmentNormalize):

    def __init__(self, keys):
        # backgroud label is 0
        seg_table = [
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43,
            44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 66, 67, 68, 81, 82,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161,
            162, 163, 164, 165, 166
        ]
        super().__init__(keys=keys, seg_table=seg_table)
