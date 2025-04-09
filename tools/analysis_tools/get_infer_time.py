# Copyright (c) MMIPT. All rights reserved.
import argparse
from functools import partial

import torch
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmipt.registry import MODELS
from mmipt.structures import DataSample


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get a model timing cost',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--task', default='reg', help='task name. [reg|seg]')
    parser.add_argument('--steps', type=int, default=50, help='total steps')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[160, 192, 224],
        help='Input shape. --shape [d] h w')
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

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    start_event.record()

    for i in range(args.steps):
        outs = model(data['inputs'])

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    time_mean = elapsed_time_ms / args.steps / 1000

    # with torch.autograd.profiler.profile(
    #         enabled=True, use_device=device, with_flops=True) as prof:
    #     outs = model(data['inputs'])
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    return dict(mean=time_mean, device=device)


def main():
    """
    Examples:

    Image Registration:
    `python tools/analysis_tools/get_infer_time.py configs/registration/transmorph/transmorph_ixi_atlas-to-scan_160x192x224.py --shape 160 192 224` # noqa

    """
    args = parse_args()
    results = inference(args)
    time_mean = results['mean']
    device = results['device']

    split_line = '=' * 30
    print(f'{split_line}\nsteps: {args.steps}\n'
          f'average time: {time_mean}\n'
          f'device: {device}\n{split_line}')


if __name__ == '__main__':
    main()
