# Copyright (c) MMIPT. All rights reserved.
from typing import Dict, List, Sequence, Tuple

from mmengine.runner.loops import EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmipt.registry import LOOPS
from .loop_utils import exchange_data


@LOOPS.register_module()
class ExchangeEpochBasedTrainLoop(EpochBasedTrainLoop):
    """Train loop for MMipt models which support exchange data in single iter.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict or list): A dataloader object or a dict
            to build a dataloader a list of dataloader object or a list of
            config dicts.
        evaluator (Evaluator or dict or list): A evaluator object or a dict to
            build the evaluator or a list of evaluator object or a list of
            config dicts.
    """

    def __init__(
            self,
            runner,
            dataloader: DataLoader | Dict,
            with_seg: bool,
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: List[Tuple[int, int]] | None = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)
        self.with_seg = with_seg

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        data_batch = exchange_data(data_batch, self.with_seg)
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
