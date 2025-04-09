# Copyright (c) MMIPT. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmipt.registry import METRICS


@METRICS.register_module()
class DiceLKUNetMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: -1.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mDice', 'HD95' and 'ASD'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        save_metric (bool): If True, save metric to csv file. Defaults to False.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = -1,
                 iou_metrics: List[str] = ['mDice'],
                 nan_to_num: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 save_metric: bool = True,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(iou_metrics, str):
            iou_metrics = [iou_metrics]

        if not set(iou_metrics).issubset({'mDice'}):
            raise KeyError(f'metrics {iou_metrics} is not supported. '
                           f'Only supports mIoU/mDice/mFscore.')

        if ignore_index not in [-1, 0]:
            raise ValueError(f'ignore_index {ignore_index} is not supported. '
                             f'Only supports [-1, 0]. '
                             f'We assume that the background label is 0.')

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        self.save_metric = save_metric
        self.format_only = format_only

        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
            # use LIA transform as default affine
            self.affine = np.array(
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=float)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_seg = data_sample['pred_seg'].squeeze().long()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                target_seg = data_sample['target_seg'].squeeze().long()
                result = self.dice_lkunet(pred_seg.cpu().detach().numpy(),
                                          target_seg.cpu().detach().numpy())
                self.results.append(result)

            # format_result
            if self.output_dir is not None:
                try:
                    import nibabel as nib
                except ImportError:
                    error_msg = (
                        '{} need to be installed! Run `pip install -r '
                        'requirements/runtime.txt` and try again')
                    raise ImportError(error_msg.format('\'nibabel\''))

                src_name = osp.splitext(
                    osp.basename(data_sample['source_img_path']))[0]
                tgt_name = osp.splitext(
                    osp.basename(data_sample['target_img_path']))[0]
                save_name = f'{src_name}_to_{tgt_name}.nii.gz'
                save_path = osp.abspath(osp.join(self.output_dir, save_name))
                pred_seg = pred_seg.cpu().numpy().astype(np.uint16)
                new_image = nib.Nifti1Image(pred_seg, self.affine)
                nib.save(new_image, save_path)

    @staticmethod
    def dice_lkunet(pred_label: np.ndarray,
                    label: np.ndarray,
                    num_classes: Optional[int] = None):
        mask_value = list(set(np.unique(pred_label)) & set(np.unique(label)))

        dices = np.zeros(len(mask_value) - 1)
        for i, cls in enumerate(mask_value[1:]):
            true_clus = label == cls
            pred_clus = pred_label == cls
            intersection = (pred_clus * true_clus).sum()
            union = pred_clus.sum() + true_clus.sum()
            dices[i] = (2.0 * intersection) / union
        return np.mean(dices)

    @staticmethod
    def dice_utsrmorph(pred_label: np.ndarray,
                       label: np.ndarray,
                       num_classes: Optional[int] = None):

        roi_value = [c for c in range(num_classes)][1:]
        dices = np.zeros(len(roi_value))
        for i, cls in enumerate(roi_value):
            true_clus = label == cls
            pred_clus = pred_label == cls
            intersection = (pred_clus * true_clus).sum()
            union = pred_clus.sum() + true_clus.sum()
            dices[i] = (2.0 * intersection) / (union + 1e-5)
        return np.mean(dices[np.nonzero(dices)])

    @staticmethod
    def dice_transmorph(pred_label: np.ndarray,
                        label: np.ndarray,
                        num_classes: Optional[int] = None):

        roi_value = [c for c in range(num_classes)][1:]
        dices = np.zeros(len(roi_value))
        for i, cls in enumerate(roi_value):
            true_clus = label == cls
            pred_clus = pred_label == cls
            intersection = (pred_clus * true_clus).sum()
            union = pred_clus.sum() + true_clus.sum()
            dices[i] = (2.0 * intersection) / (union + 1e-5)
        return np.mean(dices)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        dsc = np.nanmean(results)
        return dict(mDice=dsc)
