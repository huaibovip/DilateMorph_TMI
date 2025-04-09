# Copyright (c) MMIPT. All rights reserved.
from os.path import join

import numpy as np
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Basic hook that invoke visualizers during validation and test.

    Args:
        interval (int | dict): Visualization interval. Default: {}.
        on_train (bool): Whether to call hook during train. Default to False.
        on_val (bool): Whether to call hook during validation. Default to True.
        on_test (bool): Whether to call hook during test. Default to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 on_val=True,
                 on_test=True,
                 manual_save=False,
                 by_epoch=True):
        self._on_val = on_val
        self._on_test = on_test
        self._by_epoch = by_epoch
        self._manual_save = manual_save
        assert (by_epoch == True), "not support iters"

    @master_only
    def before_test(self, runner) -> None:
        if self._manual_save:
            self.save_dir = join(runner._log_dir, 'vis_result')
            mkdir_or_exist(self.save_dir)

    @master_only
    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs) -> None:
        """:class:`VisualizationHook` do not support visualize during
        validation.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model
        """

        if not self._on_val:
            return

        # TODO epoch-based
        if self.end_of_epoch(runner.val_dataloader, batch_idx):
            step = runner.epoch if self._by_epoch else runner.iter
            for data_sample in outputs:  # num of bacth
                runner.visualizer.add_datasample(
                    name='val', data_sample=data_sample, step=step)

    @master_only
    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs):
        """Visualize samples after test iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs: outputs of the generation model Defaults to None.
        """
        if not self._on_test:
            return

        for idx, data_sample in enumerate(outputs):  # num of bacth
            runner.visualizer.add_datasample(
                name='test', data_sample=data_sample, step=batch_idx)

            if self._manual_save:
                sidx = batch_idx * len(outputs) + idx
                mov_img = to_numpy(data_batch['inputs']['source_img'][idx])
                fix_img = to_numpy(data_batch['inputs']['target_img'][idx])
                mov_seg = to_numpy(data_batch['data_samples'][idx].source_seg)
                fix_seg = to_numpy(data_batch['data_samples'][idx].target_seg)
                np.savez(join(self.save_dir, f'mov_img_{sidx}.npz'), mov_img)
                np.savez(join(self.save_dir, f'fix_img_{sidx}.npz'), fix_img)
                np.savez(join(self.save_dir, f'mov_seg_{sidx}.npz'), mov_seg)
                np.savez(join(self.save_dir, f'fix_seg_{sidx}.npz'), fix_seg)

                pred_seg = to_numpy(data_sample.pred_seg)
                pred_img = to_numpy(data_sample.pred_img)  #
                pred_grd = to_numpy(data_sample.pred_grid)  #
                pred_flw = to_numpy(data_sample.pred_flow, False)
                np.savez(join(self.save_dir, f'pred_seg_{sidx}.npz'), pred_seg)
                np.savez(join(self.save_dir, f'pred_img_{sidx}.npz'), pred_img)
                np.savez(join(self.save_dir, f'pred_grd_{sidx}.npz'), pred_grd)
                np.savez(join(self.save_dir, f'pred_flw_{sidx}.npz'), pred_flw)


def to_numpy(data, squeeze=True):
    data = data.cpu().numpy()
    if squeeze:
        data = data.squeeze(0)
    return data
