import glob
import itertools
import json
import os
import os.path as osp
import shutil
import sys
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import torch
from mmengine.utils.path import mkdir_or_exist

cur_path = os.path.abspath(osp.dirname(__file__))
root_path = osp.abspath(osp.join(cur_path, '../../../..'))
sys.path.append(root_path)

from mmipt.utils.data_meta import get_metainfo


def download(url, dir, unzip=True, delete=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def move(data_root, splits, delete=True):
    # splits = dict(train=[0, 394], val=[394, 414])
    # from mmengine.fileio import dump
    paths = glob.glob(osp.join(data_root, 'IXI_OAS1_*'))
    for phase in splits.keys():
        os.makedirs(osp.join(data_root, phase), exist_ok=True)
        left, right = splits[phase]
        move_cfg = [
            dict(
                src=osp.join(path, 'aligned_norm.nii.gz'),
                dst=osp.join(
                    data_root, phase,
                    'img_{}.nii.gz'.format(osp.basename(path).split('_')[-2])),
            ) for path in paths[left:right]
        ]
        move_cfg += [
            dict(
                src=osp.join(path, 'aligned_seg35.nii.gz'),
                dst=osp.join(
                    data_root, phase,
                    'seg_{}.nii.gz'.format(osp.basename(path).split('_')[-2])),
            ) for path in paths[left:right]
        ]
        # dump(move_cfg, 'move.json')
        for cfg in move_cfg:
            shutil.move(cfg['src'], cfg['dst'])
    # delete
    if delete:
        delete_path = glob.glob(osp.join(data_root, 'IXI_OAS1_*'))
        for p in delete_path:
            shutil.rmtree(p)


def struct_data(paths, phase):
    if phase == 'train':
        filenames = list(itertools.permutations(paths, 2))
    else:
        filenames = list(zip(paths[1:], paths[:-1]))
    return filenames


def write_ann(file, prefix, paths):
    data_list = list()
    for i in range(len(paths)):
        source_path = f'{prefix}/{osp.basename(paths[i][0])}'
        target_path = f'{prefix}/{osp.basename(paths[i][1])}'
        data = dict(
            target_img_path=target_path,
            target_seg_path=target_path.replace('img', 'seg'),
            source_img_path=source_path,
            source_seg_path=source_path.replace('img', 'seg'))
        data_list.append(data)

    data_ann = dict()
    data_ann['metainfo'] = get_metainfo('ixi3d')
    data_ann['data_list'] = data_list
    os.makedirs(os.path.dirname(file), exist_ok=True)

    print(
        f'Generate annotation files {osp.basename(file)}({len(data_list)})...')
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data_ann, f, ensure_ascii=False, sort_keys=False, indent=4)


if __name__ == '__main__':
    # dataset: https://surfer.nmr.mgh.harvard.edu/ftp/data/ixi
    data_link = None
    root = 'data/reg/IXI'
    data_root = osp.join(root, 'datas')
    splits = dict(train=(0, 403), val=(403, 461), test=(461, 576))

    mkdir_or_exist(data_root)
    download(data_link, dir=data_root, delete=False)
    move(data_root, splits, delete=True)

    for phase in splits.keys():
        paths = glob.glob(osp.join(data_root, phase, 'img_*.nii.gz'))
        phase_paths = struct_data(paths, phase)
        write_ann(
            osp.join(root, 'annotations', f'{phase}_ixi3d_scan2scan.json'),
            prefix=f'datas/{phase}',
            paths=phase_paths)
