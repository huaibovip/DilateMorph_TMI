# Copyright (c) MMIPT. All rights reserved.
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.fileio import put_text
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable

from mmipt.registry import METRICS


@METRICS.register_module()
class DiceMetric(BaseMetric):
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

        if not set(iou_metrics).issubset({'mIoU', 'mDice', 'mFscore'}):
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

        if self.format_only:
            assert (self.output_dir is not None
                    ), 'whern \'format_only\' is set, \'output_dir\' is needed'

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
                result = self.intersect_and_union(
                    pred_seg,
                    target_seg,
                    num_classes,
                    self.ignore_index,
                )
                self.results.append(result)

            # format_result
            if self.output_dir is not None:
                dataset_type = self.dataset_meta['dataset_type']
                save_data(data_batch, data_sample, self.output_dir)
                return
                if dataset_type.find('oasis') != -1:
                    save_oasis_from_pkl(data_sample, self.output_dir)
                else:
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
                    save_path = osp.abspath(
                        osp.join(self.output_dir, save_name))
                    pred_seg = pred_seg.cpu().numpy().astype(np.uint16)
                    new_image = nib.Nifti1Image(pred_seg, self.affine)
                    nib.save(new_image, save_path)

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
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
        # mask = torch.zeros_like(label, dtype=torch.bool, device=label.device)
        # for c in torch.unique(label):
        #     mask[pred_label == c] = True
        # pred_label[mask == False] = 0

        if ignore_index == 0:
            min = 1
            bins = num_classes - 1
        else:
            min = 0
            bins = num_classes

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=bins, min=min, max=num_classes - 1)
        area_pred_label = torch.histc(
            pred_label.float(), bins=bins, min=min, max=num_classes - 1)
        area_label = torch.histc(
            label.float(), bins=bins, min=min, max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect

        result = dict(
            area_intersect=area_intersect,
            area_union=area_union,
            area_pred_label=area_pred_label,
            area_label=area_label)
        return result

    def _calculate_metrics(self, results: list) -> dict:
        """Calculate evaluation metrics.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """
        dices = []
        for result in results:
            area_intersect = result['area_intersect']
            area_union = result['area_union']
            area_pred_label = result['area_pred_label']
            area_label = result['area_label']
            # dice score
            dice = 2.0 * area_intersect / (area_pred_label + area_label)
            dice = dice.cpu().numpy()
            # nan to num
            if self.nan_to_num is not None:
                dice = np.nan_to_num(dice.items(), nan=self.nan_to_num)
            dices.append(dice)

        metric_table = np.stack(dices, axis=0)
        dice = np.nanmean(metric_table, axis=0)
        return dict(mDice=dice), metric_table

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

        metrics, metric_table = self._calculate_metrics(results)

        # if label0 is background, it will be delete (registration)
        classes = self.dataset_meta['classes']
        if self.ignore_index == 0:
            classes = classes[1:]
            print_log('metrics without background (label 0)', logger)

        # each class table
        print_semantic_table(metrics, classes, logger)

        # save to csv
        if self.save_metric:
            save_root = osp.dirname(logger.log_file)
            suffix = osp.basename(osp.dirname(save_root))
            for metric in self.metrics:
                save_path = osp.join(save_root,
                                     f'{metric.lower()}_{suffix}.csv')
                np.savetxt(
                    save_path,
                    metric_table,
                    fmt='%.9f',
                    delimiter=',',
                    header=','.join(classes))
                print_log(f'The file was saved to {save_path}', logger=logger)

        return dict(mDice=np.nanmean(metrics['mDice']))


def print_semantic_table(
        results: dict,
        class_names: list,
        logger: Optional[Union[MMLogger, str]] = None) -> None:
    """Print semantic segmentation evaluation results table.

    Args:
        results (dict): The evaluation results.
        class_names (list): Class names.
        logger (MMLogger | str, optional): Logger used for printing.
            Default: None.
    """
    # each class table
    results.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric:
        np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in results.items()
    })

    print_log('per class results:', logger)
    if PrettyTable:
        class_table_data = PrettyTable()
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        print_log('\n' + class_table_data.get_string(), logger)
    else:
        logger.warning(
            '`prettytable` is not installed, for better table format, '
            'please consider installing it with "pip install prettytable"')
        print_result = {}
        for class_name, iou, acc in zip(class_names, ret_metrics_class['IoU'],
                                        ret_metrics_class['Acc']):
            print_result[class_name] = {'IoU': iou, 'Acc': acc}
        print_log(print_result, logger)


index = 0
def save_data(data_batch, data_sample, output_dir):
    global index
    logger: MMLogger = MMLogger.get_current_instance()
    save_root = osp.dirname(logger.log_file)
    os.makedirs(osp.join(save_root, 'vis_result'), exist_ok=True)
    save_path = osp.abspath(osp.join(save_root, 'vis_result', '{}_{}.npz'))
    # mimg = data_batch['inputs']['source_img'][0].squeeze().cpu().numpy()
    # fimg = data_batch['inputs']['target_img'][0].squeeze().cpu().numpy()
    # mseg = data_batch['data_samples'][0].source_seg.squeeze().cpu().numpy()
    # fseg = data_batch['data_samples'][0].target_seg.squeeze().cpu().numpy()
    # np.savez(save_path.format('mov_img', index), mimg)
    # np.savez(save_path.format('fix_img', index), fimg)
    # np.savez(save_path.format('mov_seg', index), mseg)
    # np.savez(save_path.format('fix_seg', index), fseg)
    # save prediction
    flw = data_sample['pred_flow'].squeeze().cpu().numpy()
    grd = data_sample['pred_grid'].squeeze().cpu().numpy()
    img = data_sample['pred_img'].squeeze().cpu().numpy()
    seg = data_sample['pred_seg'].squeeze().cpu().numpy()
    np.savez(save_path.format('pred_flw', index), flw)
    np.savez(save_path.format('pred_grd', index), grd)
    np.savez(save_path.format('pred_img', index), img)
    np.savez(save_path.format('pred_seg', index), seg)
    index = index + 1


def save_oasis_data(data_sample, output_dir):
    try:
        from scipy.ndimage import zoom
    except ImportError:
        error_msg = ('{} need to be installed! Run `pip install -r '
                     'requirements/runtime.txt` and try again')
        raise ImportError(error_msg.format('\'scipy\''))
    put_text(
        'This is a dummy file required for evaluation on the Grand Challenge website.'
        ' Please DO NOT delete it.',
        osp.join(osp.dirname(output_dir), 'dummy.txt'))

    src_id = osp.basename(data_sample['source_img_path'])[4:8]
    dst_id = osp.basename(data_sample['target_img_path'])[4:8]
    save_name = f'disp_{dst_id}_{src_id}.npz'
    save_path = osp.abspath(osp.join(output_dir, save_name))

    flow = data_sample['pred_flow'].squeeze().cpu().numpy()
    flow = np.array([zoom(flow[i], 0.5, order=2)
                     for i in range(3)]).astype(np.float16)
    np.savez(save_path, flow)


def save_oasis_from_pkl(data_sample, output_dir):
    try:
        from scipy.ndimage import zoom
    except ImportError:
        error_msg = ('{} need to be installed! Run `pip install -r '
                     'requirements/runtime.txt` and try again')
        raise ImportError(error_msg.format('\'scipy\''))
    put_text(
        'This is a dummy file required for evaluation on the Grand Challenge website.'
        ' Please DO NOT delete it.',
        osp.join(osp.dirname(output_dir), 'dummy.txt'))

    src_name = osp.splitext(osp.basename(data_sample['source_img_path']))[0]
    save_name = src_name.replace('p_', 'disp_') + '.npz'
    save_path = osp.abspath(osp.join(output_dir, save_name))

    flow = data_sample['pred_flow'].squeeze().cpu().numpy()
    flow = np.array([zoom(flow[i], 0.5, order=2)
                     for i in range(3)]).astype(np.float16)
    np.savez(save_path, flow)
