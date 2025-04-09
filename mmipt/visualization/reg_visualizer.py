# Copyright (c) MMIPT. All rights reserved.
import logging
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.visualization import Visualizer

from mmipt.registry import VISUALIZERS
from mmipt.structures import DataSample
from mmipt.utils import print_colored_log


@VISUALIZERS.register_module()
class RegVisualizer(Visualizer):
    r"""Registration Visualizer.

    Image to be visualized can be:
        - torch.Tensor or np.array
        - Image sequences of shape (T, C, H, W)
        - Multi-channel image of shape (1/3, H, W)
        - Single-channel image of shape (C, H, W)

    Args:
        fn_key (str): key used to determine file name for saving image.
            Usually it is the path of some input image. If the value is
            `dir/basename.ext`, the name used for saving will be basename.
        img_keys (str): keys, values of which are images to visualize.
        pixel_range (dict): min and max pixel value used to denormalize images,
            note that only float array or tensor will be denormalized,
            uint8 arrays are assumed to be unnormalized.
        bgr2rgb (bool): whether to convert the image from BGR to RGB.
        name (str): name of visualizer. Default: 'visualizer'.
        *args and \**kwargs: Other arguments are passed to `Visualizer`. # noqa
    """

    def __init__(
        self,
        img_keys: Sequence[str],
        pixel_range=dict(pred_grid=(0., 1.)),
        bgr2rgb=False,
        alpha: float = 0.8,
        name: str = 'visualizer',
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name, *args, **kwargs)
        self.img_keys = img_keys
        self.pixel_range = pixel_range
        self.bgr2rgb = bgr2rgb
        self.alpha = alpha

    def _try_get_image_or_seg(self, img: np.ndarray, key: str) -> np.ndarray:
        """from volume get image or seg.

        Returns:
            seg: (1, H, W)
            scan: (H, W, 3)
            grid: (H, W, 3)
            flow: (H, W, 3)
        """

        def make_grid(img: np.ndarray) -> np.ndarray:
            # (C,H,W,8) to (2*H,4*W,C)
            img = np.rot90(img, k=-1, axes=[1, 2])
            mid = img.shape[1] // 2
            img = img[..., mid:mid + 8]
            channel, height, width, _ = img.shape
            shape1 = (channel, height, width, 2, 4)
            shape2 = (2 * height, 4 * width, channel)
            img = img.reshape(shape1).transpose(3, 1, 4, 2, 0).reshape(shape2)
            return img

        if key.find('seg') != -1:
            # Get middle slice
            if img.ndim == 4:
                # (1,D,H,W) to (1,H,W)
                img = make_grid(img)
                img = np.transpose(img, axes=(2, 0, 1))
        elif key.find('flow') != -1:
            # Get middle slice
            if img.ndim == 4:
                # (3,D,H,W) to (H,W,3)
                img = make_grid(img)
            elif img.ndim == 3 and img.shape[0] == 2:
                # (2,H,W) to 2x(H,W,1)
                # img = np.concatenate([img[0], img[1]], axis=1)
                img = img[0][..., None]
        else:  # grid or scan
            # Get volume middle slice
            if img.ndim == 4:
                # (1,D,H,W) to (H,W,1)
                img = make_grid(img)
            # Gray image to 3 channel
            if img.ndim == 3 and img.shape[0] == 1:
                img = np.stack((img[0], img[0], img[0]), axis=2)
            # Gray image to 3 channel
            if img.ndim == 3 and img.shape[2] == 1:
                img = np.concatenate((img, img, img), axis=2)

        return img

    def _draw_seg(
        self,
        sem_seg: np.ndarray,
        classes: List,
        palette: List,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions. (1,H,W)
            classes (list): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
            palette (list): Input palette for result rendering, which
                is a list of color palette responding to the classes.
            image (np.ndarray, optional): The image to draw. (H,W,3).
                Defaults to None.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        img_size = (*sem_seg.shape[-2:], 3)
        seg = np.zeros(img_size, dtype=np.uint8)
        for label, color in zip(labels, colors):
            seg[sem_seg[0] == label, :] = color

        if image:
            seg = (image * (1 - self.alpha) + seg * self.alpha).astype(
                np.uint8)

        return seg

    def add_datasample(self,
                       name: str,
                       data_sample: DataSample,
                       step=0) -> None:
        """Concatenate image and draw.

        Args:
            input (torch.Tensor): Single input tensor from data_batch.
            data_sample (DataSample): Single data_sample from data_batch.
            output (DataSample): Single prediction output by model.
            step (int): Global step value to record. Default: 0.
        """
        prefix = '' if name is None else name + '_'

        merged_dict = {
            **data_sample.to_dict(),
        }

        if 'output' in merged_dict.keys():
            merged_dict.update(**merged_dict['output'])

        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        img_dict = {}
        for k in self.img_keys:
            if k not in merged_dict:
                print_colored_log(
                    f'Key "{k}" not in data_sample or outputs',
                    level=logging.WARN)
                continue

            img = merged_dict[k]

            # PixelData
            if isinstance(img, dict) and ('data' in img):
                img = img['data']

            # Tensor to array
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            img = self._try_get_image_or_seg(img, k)

            if k.find('seg') != -1:
                img = self._draw_seg(img, classes, palette)
                img_dict.update({k: img})
                continue

            if self.bgr2rgb:
                img = img[..., ::-1]

            # flow and grid
            if img.dtype != np.uint8:
                # We assume uint8 type are not normalized
                if k in self.pixel_range:
                    min_, max_ = self.pixel_range.get(k)
                else:
                    min_, max_ = img.min(), img.max()

                img = ((img - min_) / (max_ - min_)) * 255
                img = img.clip(0, 255).round().astype(np.uint8)

            img_dict.update({k: img})

        for key, img in img_dict.items():
            for vis_backend in self._vis_backends.values():
                vis_backend.add_image(prefix + key, img, step)
