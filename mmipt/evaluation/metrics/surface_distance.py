# Copyright (c) MMIPT. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist

from mmipt.registry import METRICS
from ..functional import surface_distance as sd


@METRICS.register_module()
class SurfaceDistanceMetric(BaseMetric):
    """surface distance evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: -1.
        metrics (list[str] | str): Metrics to be calculated, the options
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
                 percent: int = 95,
                 metrics: List[str] = ['HD', 'ASD'],
                 nan_to_num: Optional[int] = None,
                 output_dir: Optional[str] = None,
                 save_metric: bool = True,
                 format_only: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(metrics, str):
            metrics = [metrics]

        if not set(metrics).issubset({'HD', 'ASD'}):
            raise KeyError(f'metrics {metrics} is not supported. '
                           f'Only supports HD/ASD.')

        if ignore_index not in [-1, 0]:
            raise ValueError(f'ignore_index {ignore_index} is not supported. '
                             f'Only supports [-1, 0]. '
                             f'We assume that the background label is 0.')

        self.ignore_index = ignore_index
        self.metrics = metrics
        self.nan_to_num = nan_to_num
        self.output_dir = output_dir
        self.save_metric = save_metric
        self.format_only = format_only
        self.percent = percent
        self.spacing_mm = (1, 1, 1)

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
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_seg = data_sample['pred_seg'].squeeze().long()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                target_seg = data_sample['target_seg'].squeeze().long()
                result = self.compute_numpy(
                    pred_seg.cpu().numpy(),
                    target_seg.cpu().numpy(),
                    metrics=self.metrics,
                    num_classes=num_classes,
                    ignore_index=self.ignore_index,
                    percent=self.percent,
                    spacing_mm=self.spacing_mm,
                )
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

        metric_table = dict()
        for metric in self.metrics:
            m = [result[metric] for result in results]
            if metric == 'HD':
                metric = metric + str(self.percent)
            metric_table.update({metric: np.stack(m, axis=0)})

        # if label0 is background, it will be delete (registration)
        classes = self.dataset_meta['classes']
        if self.ignore_index == 0:
            classes = classes[1:]
            print_log('metrics without background (label 0)', logger)

        # save to csv
        if self.save_metric:
            save_root = osp.dirname(logger.log_file)
            suffix = osp.basename(osp.dirname(save_root))
            for metric in metric_table.keys():
                save_path = osp.join(save_root,
                                     f'{metric.lower()}_{suffix}.csv')
                np.savetxt(
                    save_path,
                    metric_table[metric],
                    fmt='%.9f',
                    delimiter=',',
                    header=','.join(classes))
                print_log(f'The file was saved to {save_path}', logger=logger)

        ret_metrics = dict()
        for key, val in metric_table.items():
            ret_metrics[key] = np.nanmean(val)

        return ret_metrics

    @staticmethod
    def compute_numpy(pred_label: torch.tensor,
                      label: torch.tensor,
                      metrics: List[str],
                      num_classes: int,
                      ignore_index: int,
                      percent: int = 95,
                      spacing_mm: Sequence[int] = (1, 1, 1)):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int | list[int]): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        voi_labels = [i for i in range(num_classes)]
        if ignore_index == 0:
            # mask = (label != ignore_index)
            # label, pred_label = label[mask], pred_label[mask]
            voi_labels = voi_labels[1:]

        hds = []
        asds = []
        for i in voi_labels:
            pred_i = pred_label == i
            true_i = label == i
            if (pred_i.sum() == 0) or (true_i.sum() == 0):
                continue

            dist = sd.compute_surface_distances(true_i, pred_i, spacing_mm)
            if 'HD' in metrics:
                hds.append(sd.compute_robust_hausdorff(dist, percent))

            if 'ASD' in metrics:
                asds.append(sd.compute_average_surface_distance(dist)[0])

        result = dict()
        if len(hds) > 0:
            hd_table = np.stack(hds, axis=0)
            result['HD'] = hd_table

        if len(asds) > 0:
            asd_table = np.stack(asds, axis=0)
            result['ASD'] = asd_table

        return result
