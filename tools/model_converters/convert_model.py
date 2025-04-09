# Copyright (c) MMIPT. All rights reserved.
import argparse
from collections import OrderedDict

import torch
from packaging import version


def parse_args():
    parser = argparse.ArgumentParser(description='Process a checkpoint')
    parser.add_argument('--src_path', help='input checkpoint path')
    parser.add_argument('--dst_path', help='output checkpoint path')
    parser.add_argument('--name', default='default', help='model name')
    parser.add_argument(
        '--start', default=0, type=int, help='replace start index')
    args = parser.parse_args()
    return args


def delete_keys_func(state_dict, delete_keys):
    # remove delete_keys for smaller file size
    for k in list(state_dict.keys()):
        for delete_key in delete_keys:
            if k.find(delete_key) != -1:
                del state_dict[k]


def replace_keys_func(state_dict, replace_keys, start=0):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        flag = True
        for pre_key, post_key in replace_keys.items():
            if k.find(pre_key) != -1:
                new_k = k.replace(pre_key, post_key)
                new_state_dict[new_k] = v
                flag = False
        if flag:
            new_state_dict['backbone.' + k[start:]] = v
    return new_state_dict


def convert_state_dict(state_dict, delete_keys, replace_keys, name, start=0):
    delete_key = delete_keys[name] \
        if name in delete_keys else delete_keys['default']
    replace_key = replace_keys[name] \
        if name in replace_keys else replace_keys['default']

    delete_keys_func(state_dict, delete_key)
    return replace_keys_func(state_dict, replace_key, start)


def save_checkpoint(checkpoint, path):
    if version.parse(torch.__version__) >= version.parse('1.6'):
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, path)


def print_state_dict(state_dict):
    for k, v in state_dict.items():
        print(f'{k}: {v.shape}')


if __name__ == '__main__':
    args = parse_args()

    # start=4 for transmorph_bayes
    args.start = 0
    args.name = 'transmorph'
    args.src_path = None
    args.dst_path = 'transmorph-diff_ixi.pth'

    delete_keys = dict(
        default=['spatial_trans'],
        vxmdense=['transformer.grid'],
        vitvnet=['spatial_trans'],
        xmorpher=['spatial_trans'],
        utsrmorph=['spatial_trans'],
        transmatch=['spatial_trans'],
        transmorph_diff=[],
        transmorph=['spatial_trans'],
        transmorph_oasis=['spatial_trans', 'c2.0'],
        transmorph_dca_oasis=['spatial_trans'],
        transmorph_tvf_oasis=['spatial_trans'],
        transmorph_large_oasis=['spatial_trans', 'c2.0'],
        transmorph_large=['spatial_trans', 'c2.0'],
        transmorph_bspl=['spatial_trans', 'c2.0'],
        transmorph_bayes=['spatial_trans', 'net.reg_head', 'net.sigma_head'],
        symtrans=[],
        pivit=['transformer'],
        midir=[],
        pvt=['spatial_trans'],
        fouriernet=[
            'r_dc5', 'r_dc6', 'r_dc7', 'r_dc8', 'r_up3', 'r_up4', 'i_dc',
            'i_up'
        ],
        lkunet=['dc10'],
        dilatemorph_abmrct=[],
    )
    replace_keys = dict(
        default={'reg_head.0': 'flow.conv'},
        vxmdense={'flow': 'flow.conv'},
        vitvnet={'reg_head.0': 'flow.conv'},
        xmorpher={
            'reg_head.0': 'flow.conv',
            'swin': 'backbone'
        },
        utsrmorph={'reg_head.0': 'backbone.flow.conv'},
        transmatch={'reg_head.0': 'flow.conv'},
        transmorph_diff={},
        transmorph={'reg_head.0': 'flow.conv'},
        transmorph_bayes={'reg_head.0': 'flow.conv'},
        transmorph_large={'reg_head.0': 'flow.conv'},
        transmorph_oasis={'reg_head.0': 'flow.transforms.0.conv'},
        transmorph_dca_oasis={},
        transmorph_large_oasis={'reg_head.0': 'flow.transforms.0.conv'},
        transmorph_bspl={
            'resize_conv': 'flow.transforms.0.resize_conv',
            'out_layer': 'flow.transforms.0.conv',
        },
        transmorph_tvf_oasis={
            f'reg_heads.{i}.0': f'backbone.reg_heads.{i}.conv'
            for i in range(0, 12)
        },
        midir={
            'resize_conv': 'flow.transforms.0.resize_conv',
            'out_layer': 'flow.transforms.0.conv',
        },
        symtrans={'RegTran': 'backbone'},
        pivit={},
        pvt={'reg_head.0': 'flow.conv'},
        fouriernet={},
        lkunet={},
        dilatemorph_abmrct={'flow.conv': 'flow.conv'},
    )

    ckpt = torch.load(args.src_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    print_state_dict(state_dict)

    new_state_dict = convert_state_dict(
        state_dict=state_dict,
        delete_keys=delete_keys,
        replace_keys=replace_keys,
        name=args.name,
        start=args.start,
    )
    ckpt['state_dict'] = new_state_dict
    save_checkpoint(ckpt, args.dst_path)
    print_state_dict(new_state_dict)
