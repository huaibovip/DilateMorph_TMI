# Copyright (c) MMIPT. All rights reserved.
import argparse
from functools import partial

import torch
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmipt.registry import MODELS
from mmipt.structures import DataSample

try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get a model complexity',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--task', default='reg', help='task name. [reg|seg]')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[160, 192, 224],
        help='Input shape. --shape [d] h w')
    parser.add_argument(
        '--out-table',
        action='store_true',
        help='Whether to show the complexity table')
    parser.add_argument(
        '--out-arch',
        action='store_true',
        help='Whether to show the complexity arch')
    args = parser.parse_args()
    return args


def construct_data(shape, device, task='reg') -> tuple:
    if task == 'reg':
        inputs = [{
            'source_img': torch.randn(1, *shape, device=device),
            'target_img': torch.randn(1, *shape, device=device)
        }]
        data_samples = [DataSample(metainfo={})]
    elif task == 'seg':
        inputs = dict(img=torch.randn(1, 1, *shape, device=device))
        data_samples = [DataSample(metainfo={})]
    else:
        raise ValueError('invalid task name')

    data_batch = {
        'inputs': inputs,
        'data_samples': data_samples,
    }
    return data_batch


@torch.no_grad()
def inference(args: argparse.Namespace) -> dict:
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmipt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    model.to(device)
    model.eval()

    if hasattr(model, 'extract_feats'):
        model.forward = partial(model.extract_feats, training=False)
    if hasattr(model, 'head'):
        delattr(model, 'head')

    input_shape = tuple(args.shape)
    data_batch = construct_data(input_shape, device, args.task)
    data = model.data_preprocessor(data_batch)
    if args.task == 'reg':
        data = tuple([data['inputs']])

    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=data,
        show_table=args.out_table,
        show_arch=args.out_arch)
    outputs['compute_type'] = 'direct: randomly generate an input'

    return outputs


def main():
    """
    Examples:

    Image Registration:
    `python tools/analysis_tools/get_flops.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py --shape 160 192 224` # noqa

    """
    args = parse_args()
    results = inference(args)
    flops = results['flops_str']
    params = results['params_str']
    compute_type = results['compute_type']

    split_line = '=' * 30
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {tuple(args.shape)}\n'
          f'Flops: {flops}\n'
          f'Params: {params}\n{split_line}')
    if args.out_table:
        print(results['out_table'], '\n')
    if args.out_arch:
        print(results['out_arch'], '\n')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
