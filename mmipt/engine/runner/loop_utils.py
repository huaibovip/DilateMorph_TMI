# Copyright (c) MMIPT. All rights reserved.
from logging import WARNING
from typing import Any, Dict, List, Union
from copy import deepcopy

from mmengine import is_list_of, print_log
from mmengine.evaluator import Evaluator

EVALUATOR_TYPE = Union[Evaluator, Dict, List]


def update_and_check_evaluator(
        evaluator: EVALUATOR_TYPE) -> Union[Evaluator, dict]:
    """Check the whether the evaluator instance or dict config is Evaluator. If
    input is a dict config, attempt to set evaluator type as Evaluator and
    raised warning if it is not allowed. If input is a Evaluator instance,
    check whether it is a Evaluator class, otherwise,

    Args:
        evaluator (Union[Evaluator, dict, list]): The evaluator instance or
            config dict.
    """
    # check Evaluator instance
    warning_template = ('Evaluator type for current config is \'{}\'. '
                        'If you want to use MultiValLoop, we strongly '
                        'recommend you to use \'Evaluator\' provided by '
                        '\'MMipt\'. Otherwise, there maybe some potential '
                        'bugs.')
    if isinstance(evaluator, Evaluator):
        cls_name = evaluator.__class__.__name__
        if cls_name != 'Evaluator':
            print_log(warning_template.format(cls_name), 'current', WARNING)
        return evaluator

    # add type for **single evaluator with list of metrics**
    if isinstance(evaluator, list):
        evaluator = dict(type='Evaluator', metrics=evaluator)
        return evaluator

    # check and update dict config
    assert isinstance(evaluator, dict), (
        'Can only conduct check and update for list of metrics, a config dict '
        f'or a Evaluator object. But receives {type(evaluator)}.')
    evaluator.setdefault('type', 'Evaluator')
    evaluator.setdefault('metrics', None)  # default as 'dummy evaluator'
    _type = evaluator['type']
    if _type != 'Evaluator':
        print_log(warning_template.format(_type), 'current', WARNING)
    return evaluator


def is_evaluator(evaluator: Any) -> bool:
    """Check whether the input is a valid evaluator config or Evaluator object.

    Args:
        evaluator (Any): The input to check.

    Returns:
        bool: Whether the input is a valid evaluator config or Evaluator
            object.
    """
    # Single evaluator with type
    if isinstance(evaluator, dict) and 'metrics' in evaluator:
        return True
    # Single evaluator without type
    elif (is_list_of(evaluator, dict)
          and all(['metrics' not in cfg_ for cfg_ in evaluator])):
        return True
    elif isinstance(evaluator, Evaluator):
        return True
    else:
        return False


def exchange_data(data_batch, with_seg):
    new_data_batch = deepcopy(data_batch)
    num_batch = len(data_batch['inputs']['source_img'])

    for i in range(num_batch):
        new_data_batch['inputs']['source_img'][i] = \
            data_batch['inputs']['target_img'][i]
        new_data_batch['inputs']['target_img'][i] = \
            data_batch['inputs']['source_img'][i]
        # new_data_batch['data_samples'][i].source_img_path = \
        #     data_batch['data_samples'][i].target_img_path
        # new_data_batch['data_samples'][i].target_img_path = \
        #     data_batch['data_samples'][i].source_img_path

        if with_seg:
            new_data_batch['data_samples'][i].source_seg = \
                data_batch['data_samples'][i].target_seg
            new_data_batch['data_samples'][i].target_seg = \
                data_batch['data_samples'][i].source_seg

    return new_data_batch


def exchange_data_fast(data_batch, with_seg):
    num_batch = len(data_batch['inputs']['source_img'])

    for i in range(num_batch):
        tmp_src = data_batch['inputs']['source_img'][i]
        data_batch['inputs']['source_img'][i] = \
            data_batch['inputs']['target_img'][i]
        data_batch['inputs']['target_img'][i] = tmp_src
        # tmp_src_path = data_batch['data_samples'][i].source_img_path
        # data_batch['data_samples'][i].source_img_path = \
        #     data_batch['data_samples'][i].target_img_path
        # data_batch['data_samples'][i].target_img_path = tmp_src_path

        if with_seg:
            tmp_src_seg = data_batch['data_samples'][i].source_seg
            data_batch['data_samples'][i].source_seg = \
                data_batch['data_samples'][i].target_seg
            data_batch['data_samples'][i].target_seg = tmp_src_seg

    return data_batch
