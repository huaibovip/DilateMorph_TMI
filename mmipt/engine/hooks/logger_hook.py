# Copyright (c) MMIPT. All rights reserved.
from pathlib import Path
from typing import Dict, Optional, Union

from mmengine.hooks import LoggerHook as BaseLoggerHook
from mmengine.hooks.logger_hook import DATA_BATCH, SUFFIX_TYPE

from mmipt.registry import HOOKS


@HOOKS.register_module()
class LoggerHook(BaseLoggerHook):
    """LoggerHook inherits from :class:`mmengine.hooks.LoggerHook` and
    overwrites :meth:`self.after_train_iter` and :meth:`self.after_val_epoch`.

    This hooks removes `time`, `data_time`, `epoch` and `iter` for visualizer.
    """

    priority = 'BELOW_NORMAL'

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = True,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[Union[str, Path]] = None,
                 out_suffix: SUFFIX_TYPE = ('.json', '.log', '.py', 'yaml'),
                 keep_local: bool = True,
                 file_client_args: Optional[dict] = None,
                 log_metric_by_epoch: bool = True,
                 backend_args: Optional[dict] = None):
        super().__init__(interval, ignore_last, interval_exp_name, out_dir,
                         out_suffix, keep_local, file_client_args,
                         log_metric_by_epoch, backend_args)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and (not self.ignore_last
                   or len(runner.train_dataloader) <= self.interval)):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return

        # remove time and data_time for visualizer
        for key in ['train/time', 'train/data_time', 'epoch', 'iter']:
            if key in tag:
                tag.pop(key)

        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)

        # remove time and data_time for visualizer
        for key in ['val/time', 'val/data_time']:
            if key in tag:
                tag.pop(key)

        if self.log_metric_by_epoch:
            # Accessing the epoch attribute of the runner will trigger
            # the construction of the train_loop. Therefore, to avoid
            # triggering the construction of the train_loop during
            # validation, check before accessing the epoch.
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch

            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                iter = 0
            else:
                iter = runner.iter

            runner.visualizer.add_scalars(
                tag, step=iter, file_path=self.json_log_path)
