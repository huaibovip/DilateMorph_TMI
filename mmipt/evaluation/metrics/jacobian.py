# Copyright (c) MMIPT. All rights reserved.
from functools import lru_cache
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import correlate
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist

from mmipt.registry import METRICS


@METRICS.register_module()
class JacobianMetric(BaseMetric):
    """Jacobian evaluation metric.

    Args:
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
                 metrics: List[str] = ['npj'],
                 percent: bool = True,
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

        if not set(metrics).issubset({'npj', 'sdlogj'}):
            raise KeyError(f'metrics {metrics} is not supported. '
                           f'Only supports "npj" and "sdlogj".')

        self.metrics = metrics
        self.percent = 100 if percent else 1
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
        metric_func = dict(npj=self.jacobian_npj, sdlogj=self.jacobian_sdlogj)
        for data_sample in data_samples:
            # format_only always for test dataset without ground truth
            if not self.format_only:
                pred_flow = data_sample['pred_flow']
                result = dict()
                for name in self.metrics:
                    result.update(metric_func[name](pred_flow, self.percent))
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
                save_name = f'{src_name}_to_{tgt_name}_flow.nii.gz'
                save_path = osp.abspath(osp.join(self.output_dir, save_name))
                pred_flow = pred_flow.cpu().numpy().astype(np.uint16)
                new_image = nib.Nifti1Image(pred_flow, self.affine)
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

        ret = dict()
        for name in self.metrics:
            res = [result[name] for result in results]
            res = np.stack(res, axis=0)
            ret.update({
                f'{name.title()}': np.nanmean(res),
                f'{name.title()}_std': np.nanstd(res),
            })

            # save to csv
            if self.save_metric:
                save_root = osp.dirname(logger.log_file)
                suffix = osp.basename(osp.dirname(save_root))
                save_path = osp.join(save_root, f'{name.lower()}_{suffix}.csv')
                np.savetxt(
                    save_path, res, fmt='%.9f', delimiter=',', header=name)
                print_log(f'The file was saved to {save_path}', logger=logger)

        return ret

    @staticmethod
    def jacobian_npj(flow: torch.Tensor,
                     percent: float = 100.) -> Dict[str, float]:
        """Calculate jacobian determinant of a displacement field.

        Args:
            flow (torch.Tensor): 2D or 3D displacement field of size [nb_dims, *vol_shape],
            where vol_shape is of len nb_dims

        Returns:
            Dict[str,np.ndarray]: Metrics.
        """

        vol_shape = flow.shape[1:]
        ndims = len(vol_shape)

        assert ndims in (2, 3), "flow has to be 2D or 3D"

        # compute grid
        grid = identity(vol_shape, flow.device)

        # compute gradients
        dim = [i + 1 for i in range(ndims)]
        J = torch.gradient(flow + grid, dim=dim)

        # compute jacobian components
        if ndims == 3:
            dx, dy, dz = J[0], J[1], J[2]
            Jac0 = dx[0] * (dy[1] * dz[2] - dy[2] * dz[1])
            Jac1 = dx[1] * (dy[0] * dz[2] - dy[2] * dz[0])
            Jac2 = dx[2] * (dy[0] * dz[1] - dy[1] * dz[0])
            jac_dets = Jac0 - Jac1 + Jac2

        elif ndims == 2:
            dfdx, dfdy = J[0], J[1]
            jac_dets = dfdx[0] * dfdy[1] - dfdy[0] * dfdx[1]

        jac_dets = jac_dets.cpu().detach().numpy()

        result = dict(
            npj=np.sum(jac_dets <= 0) / np.prod(vol_shape) * percent,
            # npj_negative=np.mean(jac_dets <= 0),
            # npj_mean=np.mean(jac_dets),
            # npj_var=np.var(jac_dets),
        )

        return result

    @staticmethod
    def jacobian_sdlogj(flow: torch.Tensor,
                        percent: float = 1.) -> Dict[str, float]:
        """ Calculate the standard deviation of the logarithm of the 
            Jacobian determinant (sdlogj) of the displacement field.

        Args:
            flow (torch.Tensor): 2D or 3D displacement field of size [nb_dims, *vol_shape],
            where vol_shape is of len nb_dims

        Returns:
            Dict[str,np.ndarray]: Metrics.
        """
        flow = flow.cpu().detach().numpy()[np.newaxis, ...]
        vol_shape = flow.shape[2:]
        ndims = len(vol_shape)

        assert ndims == 3, "flow has to be 3D"

        # compute gradients
        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
        gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

        # yapf: disable
        kwargs = dict(mode='constant', cval=0.0)
        gradx_flow = np.stack([
            correlate(flow[:, 0], gradx, **kwargs),
            correlate(flow[:, 1], gradx, **kwargs),
            correlate(flow[:, 2], gradx, **kwargs)
        ], axis=1)

        grady_flow = np.stack([
            correlate(flow[:, 0], grady, **kwargs),
            correlate(flow[:, 1], grady, **kwargs),
            correlate(flow[:, 2], grady, **kwargs)
        ], axis=1)

        gradz_flow = np.stack([
            correlate(flow[:, 0], gradz, **kwargs),
            correlate(flow[:, 1], gradz, **kwargs),
            correlate(flow[:, 2], gradz, **kwargs)
        ], axis=1)
        # yapf: enable

        grad_flow = np.concatenate([gradx_flow, grady_flow, gradz_flow], 0)

        # compute sdlogj
        jacobian = grad_flow + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
        # yapf: disable
        jac_det = (
            jacobian[0, 0] * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1]) -
            jacobian[1, 0] * (jacobian[0, 1] * jacobian[2, 2] - jacobian[0, 2] * jacobian[2, 1]) +
            jacobian[2, 0] * (jacobian[0, 1] * jacobian[1, 2] - jacobian[0, 2] * jacobian[1, 1]))
        # yapf: enable

        # return
        log_jac_det = np.log((jac_det + 3).clip(1e-9, 1e9))
        result = dict(sdlogj=log_jac_det.std())

        return result


@lru_cache
def identity(shape, device) -> torch.Tensor:
    # create identity grid
    vectors = [torch.arange(0, s) for s in shape]
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grids = torch.meshgrid(vectors, indexing='ij')
    else:
        grids = torch.meshgrid(vectors)
    # z, y, x
    grid = torch.stack(grids, dim=0)
    grid = grid.float().to(device)
    return grid
