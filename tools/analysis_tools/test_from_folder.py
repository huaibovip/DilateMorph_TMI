# Copyright (c) MMIPT. All rights reserved.
import argparse
import os.path as osp
import time

from mmengine.evaluator import Evaluator
from mmengine.fileio import dump, list_dir_or_file, load
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmipt.utils import print_colored_log
from mmipt.utils.data_meta import get_metainfo
from mmipt.utils.setup_env import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('--data-root', default='data/test_folder')
    parser.add_argument('--data-type', default='ixi_30')
    parser.add_argument(
        '--experiment-name', default='transmorph_ixi_atlas-to-scan')
    parser.add_argument('--work-dir', default='./work_dirs')
    args = parser.parse_args()
    # others
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    args.timestamp = timestamp
    args.work_dir = osp.join(args.work_dir, args.experiment_name)
    args.log_dir = osp.join(args.work_dir, timestamp)
    return args


def build_logger(args,
                 log_level: str = 'INFO',
                 log_file: str = None,
                 **kwargs) -> MMLogger:
    mkdir_or_exist(args.log_dir)
    if log_file is None:
        log_file = osp.join(args.log_dir, f'{args.timestamp}.log')
    log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
    log_cfg.setdefault('name', args.experiment_name)
    log_cfg.setdefault('file_mode', 'a')
    return MMLogger.get_instance(**log_cfg)


def load_data(root):
    """ Dict of Tensor
    Keys:
        source_img_path, target_img_path,
        pred_flow, pred_seg, target_seg
    Optional Keys:
	    interp, sample_idx, num_classes, source_shape,
        target_shape, pred_grid, source_seg
    """
    data_samples = []
    for filename in list_dir_or_file(root):
        data_samples.append(load(osp.join(root, filename)))
    return None, data_samples


def eval(root, metrics, dataset_meta):
    evaluator = Evaluator(metrics=metrics)
    evaluator.dataset_meta = dataset_meta

    data, data_samples = load_data(root)
    # chunk_size indicates the number of samples processed at a time,
    # which can be adjusted according to the memory size
    results = evaluator.offline_evaluate(
        data_samples=data_samples, data=data, chunk_size=128)
    return results


def main():
    args = parse_args()
    logger = build_logger(args)
    print_colored_log(f'Working directory: {args.work_dir}')
    print_colored_log(f'Log directory: {args.log_dir}')

    register_all_modules()

    # metric info
    metrics = [
        dict(
            type='DiceLKUNetMetric',
            iou_metrics=['mDice'],
            ignore_index=0,
            output_dir=None,
            save_metric=True),
        dict(type='JacobianMetric', metrics=['npj'], output_dir=None),
        dict(type='SurfaceDistanceMetric', ignore_index=0, output_dir=None),
    ]

    # eval
    results = eval(args.data_root, metrics, get_metainfo(args.data_type))
    dump(results, osp.join(args.log_dir, f'{args.timestamp}.json'))
    logger.info(results)


if __name__ == '__main__':
    main()
