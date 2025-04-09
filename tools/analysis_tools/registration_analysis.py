import argparse
import os
import os.path as osp
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

outstructs_ixi_mapping = {
    'Brain-Stem': 'Brain-Stem',
    'Thalamus': 'Thalamus',
    'Cerebellum-Cortex': 'CC',
    'Cerebral-White-Matter': 'CWM',
    'Cerebellum-White-Matter': 'CeWM',
    'Putamen': 'Putamen',
    # 'Ventral': 'VentralDC',  # For compatibility
    'Ventral': 'Ventral',
    'Pallidum': 'Pallidum',
    'Caudate': 'Caudate',
    'Lateral-Ventricle': 'LV',
    'Hippocampus': 'Hippocampus',
    '3rd-Ventricle': '3rd-Ventricle',
    '4th-Ventricle': '4th-Ventricle',
    'Amygdala': 'Amygdala',
    'Cerebral-Cortex': 'CeCo',
    'CSF': 'CSF',
    'Choroid-Plexus': 'CP',
}

outstructs_oasis_mapping = {
    'Brain-Stem': 'Brain-Stem',
    'Thalamus': 'Thalamus',
    'Cerebellum-Cortex': 'CC',
    'Cerebral-White-Matter': 'CWM',
    'Cerebellum-White-Matter': 'CeWM',
    'Putamen': 'Putamen',
    # 'Ventral': 'VentralDC',  # For compatibility
    'Ventral': 'Ventral',
    'Pallidum': 'Pallidum',
    'Caudate': 'Caudate',
    'Lateral-Ventricle': 'LV',
    'Hippocampus': 'Hippocampus',
    '3rd-Ventricle': '3rd-Ventricle',
    '4th-Ventricle': '4th-Ventricle',
    'Amygdala': 'Amygdala',
    'Cerebral-Cortex': 'CeCo',
    'Choroid-Plexus': 'CP',
    # extra
    'Accumbens': 'Accumbens',
    'Inf-Lat-Ventricle': 'ILV',
    'Vessel': 'Vessel',
}

outstructs_lpba_mapping = {
    'Frontal Lobe': [i for i in range(14)],
    'Parietal Lobe': [i for i in range(14, 24)],
    'Occipital Lobe': [i for i in range(24, 32)],
    'Temporal Lobe': [i for i in range(32, 44)],
    'Cingulate Lobe': [i for i in range(44, 48)],
    'Caudate': [48, 49],
    'Putamen': [50, 51],
    'Hippocampus': [52, 53],
}

methods_ixi = {
    'niftyreg': 'NiftyReg',
    'syn': 'SyN',
    'vxm2': 'VoxelMorph',
    'midir': 'MIDIR',
    'lkunet': 'LKU-Net',
    'fouriernet': 'Fourier-Net',
    'vit-vnet': 'ViT-V-Net',
    'transmorph': 'TransMorph',
    'xmorpher': 'XMorpher',
    'transmatch': 'TransMatch',
    'utsrmorph': 'UTSRMorph',
    'dilatemorph3': 'DilateMorph',
}


def merge(table: pd.DataFrame, outstructs_mapping: dict):
    # only for list/tuple
    # if isinstance(next(iter(outstructs_mapping.values())), (list, tuple)):
    #     _labels = []
    #     for ids in outstructs_mapping.values():
    #         _labels.extend(ids)
    #     idxmap = {label: idx for idx, label in enumerate(sorted(_labels))}

    datas = dict()
    for stru, alias in outstructs_mapping.items():
        if isinstance(alias, str):
            # ignore case
            columns = table.columns.str.contains(stru, case=False)
            datas[alias] = table.loc[:, columns].mean(axis=1)
        elif isinstance(alias, (list, tuple)):
            # columns = [idxmap[i] for i in alias]
            columns = alias
            datas[stru] = table.iloc[:, columns].mean(axis=1)
        else:
            raise ValueError('The type of outstructs is error')
    new_table = pd.DataFrame(datas)
    return new_table


def measure(
    metric_name: str,
    exp_cfgs: dict,
    outstructs_mapping: dict = None,
    per_patient: bool = True,
):
    print(f'\n{metric_name} (mean ± std)'.title())

    # mean & std by class (axis = 0)
    # mean & std by patient (axis = 1)
    axis = 1 if per_patient else 0

    datas = dict()
    metrics = dict()
    for name, path in exp_cfgs.items():
        # read and merge
        csv_table = pd.read_csv(path)
        if outstructs_mapping is not None:
            csv_table = merge(csv_table, outstructs_mapping)
            spath = osp.join(osp.dirname(path), 'merge_' + osp.basename(path))
            csv_table.to_csv(spath, sep=',', index=0)
        # cal metric
        data_table = csv_table.to_numpy(copy=True)
        metric_per_x = np.nanmean(data_table, axis=axis)
        # save
        metrics[name] = metric_per_x
        datas[name] = data_table
        if metric_name.lower().find('dice') != -1:
            print(
                f'{name:>20}: {metric_per_x.mean() * 100:.2f} ± {metric_per_x.std() * 100:.2f}'
            )
        else:
            print(
                f'{name:>20}: {metric_per_x.mean():.3f} ± {metric_per_x.std():.3f}'
            )
    return list(csv_table), datas, metrics


def measure2(metric_name: str, exp_cfgs: dict):
    print(f'\n{metric_name} (mean ± std)'.title())

    for name, path in exp_cfgs.items():
        csv_table = pd.read_csv(path)
        data_table = csv_table.to_numpy(copy=True)
        metric_mean = np.nanmean(data_table)
        metric_std = np.nanstd(data_table)
        print(f'{name:>20}: {metric_mean:.3f} ± {metric_std:.3f}')

    return None


def ttest(metrics: dict, main: str = 'ours'):
    """ T-test """
    print(f'\nP-Value'.title())
    keys = list(metrics.keys())
    main_key = keys.pop(keys.index(main))
    main_vec = metrics[main_key]
    for name in keys:
        rank, pval = ttest_rel(list(main_vec), list(metrics[name]))
        print(f'{name:>20}: {pval:.20f}')


if __name__ == '__main__':
    """ Directory structure
    work_dirs/results:
        ├─ixi
        |  ├─transmorph
        |  │  ├─asd_transmorph_ixi_atlas-to-scan.csv
        │  |  ├─hd95_transmorph_ixi_atlas-to-scan.csv
        │  |  ├─jdet_transmorph_ixi_atlas-to-scan.csv
        │  |  ├─mdice_transmorph_ixi_atlas-to-scan.csv
        |  |  └─...
        │  ├─fouriernet
        |  └─...
        ├─lpba
        └─...
    """

    parser = argparse.ArgumentParser(description='Analysis tools')
    parser.add_argument('--metric', default='mDice')
    parser.add_argument('--dataset', default='ixi')
    parser.add_argument('--root', default='work_dirs/results')
    # parser.add_argument('--out', help='the file to save metric results.')
    args = parser.parse_args()

    metric = args.metric.lower()
    assert metric in ['mdice', 'asd', 'hd95', 'jdet', 'sdlogj']
    data_root = osp.join(args.root, args.dataset)

    exp_cfgs = dict()
    for model_name in os.listdir(data_root):
        csv_path = glob(osp.join(data_root, model_name, f'{metric}*.csv'))[0]
        exp_cfgs[model_name] = csv_path
        print('{0:>20}: {1}'.format(model_name, osp.basename(csv_path)))

    if args.metric in ['jdet', 'sdlogj']:
        measure2(metric, exp_cfgs)
    else:
        outstructs, datas, metrics = measure(
            metric,
            exp_cfgs,
            eval(f'outstructs_{args.dataset}_mapping'),
            per_patient=True,
        )
        ttest(metrics, main='dilatemorph')
