# Copyright (c) MMIPT. All rights reserved.
import argparse
import os
import os.path as osp

from matplotlib import pyplot as plt

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from mmipt.utils import print_colored_log


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model.cpu().eval()

    print_colored_log(f'Working directory: {cfg.work_dir}')
    print_colored_log(f'Log directory: {runner._log_dir}')

    with torch.no_grad():
        loader = iter(runner.test_dataloader)
        inputs = next(loader)['inputs']
        inputs = dict(
            source_img=inputs['source_img'][0].unsqueeze(0),
            target_img=inputs['target_img'][0].unsqueeze(0),
        )

        model_jit = torch.jit.trace(model, example_inputs=inputs)
        output1 = model(inputs)[0]
        output2 = model_jit(inputs)[0]
        model_jit.save('export_model.pt')
        # torch.onnx.export(model, inputs, f'export_model.onnx')

        print(output1.shape)
        print(torch.abs(output1 - output2).sum())
        plt.imshow(output1.cpu().numpy()[0, 0, 80])
        plt.savefig('flow1.png')
        plt.imshow(output2.cpu().numpy()[0, 0, 80])
        plt.savefig('flow2.png')
        print('save!')


if __name__ == '__main__':
    main()
