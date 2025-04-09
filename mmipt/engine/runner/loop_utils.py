# Copyright (c) MMIPT. All rights reserved.
from copy import deepcopy


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
