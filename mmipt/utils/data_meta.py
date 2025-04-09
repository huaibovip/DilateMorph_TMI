# Copyright (c) MMIPT. All rights reserved.
from mmengine.utils import is_str


def ixi_30_classes():
    """IXI class names for external use."""
    return [
        'Unknown', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex',
        'Left-Lateral-Ventricle', 'Left-Cerebellum-White-Matter',
        'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper*', 'Left-Caudate',
        'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
        'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
        'Left-VentralDC', 'Left-Choroid-Plexus', 'Right-Cerebral-White-Matter',
        'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle',
        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
        'Right-Thalamus-Proper*', 'Right-Caudate', 'Right-Putamen',
        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
        'Right-VentralDC', 'Right-Choroid-Plexus'
    ]


def ixi_30_palette():
    """IXI palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [220, 248, 164], [230, 148, 34], [0, 118, 14], [122, 186, 220],
            [236, 13, 176], [12, 48, 255], [204, 182, 142], [42, 204, 164],
            [119, 159, 176], [220, 216, 20], [103, 255, 255], [60, 60, 60],
            [165, 42, 42], [0, 200, 200], [245, 245, 245], [205, 62, 78],
            [120, 18, 134], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [13, 48, 255], [220, 216, 20],
            [103, 255, 255], [165, 42, 42], [0, 200, 221]]


def oasis2d_24_classes():
    """OASIS class names for external use."""
    return [
        "Unknown", "Left-Cerebral-White-Matter", "Left-Cerebral-Cortex",
        "Left-Lateral-Ventricle", "Left-Inf-Lat-Ventricle", "Left-Thalamus",
        "Left-Caudate", "Left-Putamen", "Left-Pallidum", "3rd-Ventricle",
        "Brain-Stem", "Left-Hippocampus", "Left-VentralDC",
        "Left-Choroid-Plexus", "Right-Cerebral-White-Matter",
        "Right-Cerebral-Cortex", "Right-Lateral-Ventricle",
        "Right-Inf-Lat-Ventricle", "Right-Thalamus", "Right-Caudate",
        "Right-Putamen", "Right-Pallidum", "Right-Hippocampus",
        "Right-VentralDC", "Right-Choroid-Plexus"
    ]


def oasis2d_24_palette():
    """OASIS palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [196, 58, 250], [0, 118, 14], [122, 186, 220], [236, 13, 176],
            [12, 48, 255], [204, 182, 142], [119, 159, 176], [220, 216, 20],
            [165, 42, 42], [0, 200, 200], [245, 245, 245], [205, 62, 78],
            [120, 18, 134], [196, 58, 250], [0, 118, 14], [122, 186, 220],
            [236, 13, 176], [12, 48, 255], [220, 216, 20], [165, 42, 42],
            [0, 200, 200]]


def oasis3d_4_classes():
    """OASIS class names for external use."""
    return [
        'Unknown', 'Cortex', 'Subcortical-Gray-Matter', 'White-Matter', 'CSF'
    ]


def oasis3d_4_palette():
    """OASIS palette for external use."""
    return [[0, 0, 0], [205, 62, 78], [119, 159, 176], [245, 245, 245],
            [120, 18, 134]]


def oasis3d_35_classes():
    """OASIS class names for external use."""
    return [
        'Unknown', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex',
        'Left-Lateral-Ventricle', 'Left-Inf-Lat-Ventricle',
        'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
        'Left-Thalamus-Proper*', 'Left-Caudate', 'Left-Putamen',
        'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
        'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens',
        'Left-VentralDC', 'Left-Vessel', 'Left-Choroid-Plexus',
        'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex',
        'Right-Lateral-Ventricle', 'Right-Inf-Lat-Ventricle',
        'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
        'Right-Thalamus-Proper*', 'Right-Caudate', 'Right-Putamen',
        'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
        'Right-Accumbens', 'Right-VentralDC', 'Right-Vessel',
        'Right-Choroid-Plexus'
    ]


def oasis3d_35_palette():
    """OASIS palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
            [122, 186, 220], [236, 13, 176], [12, 48, 255], [204, 182, 142],
            [42, 204, 164], [119, 159, 176], [220, 216, 20], [103, 255, 255],
            [255, 165, 0], [165, 42, 42], [160, 32, 240], [0, 200, 200],
            [245, 245, 245], [205, 62, 78], [120, 18, 134], [196, 58, 250],
            [220, 248, 164], [230, 148, 34], [0, 118, 14], [122, 186, 220],
            [236, 13, 176], [12, 48, 255], [220, 216, 20], [103, 255, 255],
            [255, 165, 0], [165, 42, 42], [160, 32, 240], [0, 200, 200]]


def lpba40_54_classes():
    """LPBA40 class names for external use."""
    return [
        'Unknown', 'L-Superior-Frontal-Gyrus', 'R-Superior-Frontal-Gyrus',
        'L-Middle-Frontal-Gyrus', 'R-Middle-Frontal-Gyrus',
        'L-Inferior-Frontal-Gyrus', 'R-Inferior-Frontal-Gyrus',
        'L-Precentral-Gyrus', 'R-Precentral-Gyrus',
        'L-Middle-Orbitofrontal-Gyrus', 'R-Middle-Orbitofrontal-Gyrus',
        'L-Lateral-Orbitofrontal-Gyrus', 'R-Lateral-Orbitofrontal-Gyrus',
        'L-Gyrus-Rectus', 'R-Gyrus-Rectus', 'L-Postcentral-Gyrus',
        'R-Postcentral-Gyrus', 'L-Superior-Parietal-Gyrus',
        'R-Superior-Parietal-Gyrus', 'L-Supramarginal-Gyrus',
        'R-Supramarginal-Gyrus', 'L-Angular-Gyrus', 'R-Angular-Gyrus',
        'L-Precuneus', 'R-Precuneus', 'L-Superior-Occipital-Gyrus',
        'R-Superior-Occipital-Gyrus', 'L-Middle-Occipital-Gyrus',
        'R-Middle-Occipital-Gyrus', 'L-Inferior-Occipital-Gyrus',
        'R-Inferior-Occipital-Gyrus', 'L-Cuneus', 'R-Cuneus',
        'L-Superior-Temporal-Gyrus', 'R-Superior-Temporal-Gyrus',
        'L-Middle-Temporal-Gyrus', 'R-Middle-Temporal-Gyrus',
        'L-Inferior-Temporal-Gyrus', 'R-Inferior-Temporal-Gyrus',
        'L-Parahippocampal-Gyrus', 'R-Parahippocampal-Gyrus',
        'L-Lingual-Gyrus', 'R-Lingual-Gyrus', 'L-Fusiform-Gyrus',
        'R-Fusiform-Gyrus', 'L-Insular-Cortex', 'R-Insular-Cortex',
        'L-Cingulate-Gyrus', 'R-Cingulate-Gyrus', 'L-Caudate', 'R-Caudate',
        'L-Putamen', 'R-Putamen', 'L-Hippocampus', 'R-Hippocampus'
    ]  # 'Cerebellum', 'Brainstem'


def lpba40_54_palette():
    """LPBA40 palette for external use."""
    return [[0, 0, 0], [255, 128, 128], [255, 128, 0], [0, 0, 255],
            [255, 128, 255], [128, 127, 0], [0, 0, 255], [0, 255, 255],
            [128, 0, 0], [255, 0, 128], [128, 128, 64], [128, 0, 255],
            [255, 128, 127], [128, 128, 255], [0, 255, 255], [0, 255, 0],
            [128, 128, 64], [0, 127, 128], [0, 255, 255], [255, 255, 0],
            [0, 128, 128], [255, 128, 255], [64, 0, 128], [128, 255, 128],
            [0, 0, 255], [127, 127, 255], [255, 127, 0], [255, 127, 128],
            [0, 127, 0], [160, 0, 0], [83, 166, 166], [255, 255, 127],
            [0, 0, 128], [128, 0, 64], [0, 128, 255], [92, 92, 237],
            [192, 128, 255], [17, 207, 255], [255, 0, 128], [105, 180, 31],
            [64, 255, 0], [255, 255, 0], [0, 255, 255], [35, 231, 216],
            [255, 0, 0], [0, 128, 255], [128, 255, 255], [64, 128, 255],
            [255, 0, 255], [0, 48, 255], [255, 0, 128], [128, 160, 48],
            [255, 128, 0], [133, 10, 31], [0, 255, 255]]


def abdomenmrct3d_4_classes():
    """AbdomenMRCT class names for external use."""
    return ['Unknown', 'Liver', 'Spleen', 'Right-Kidney', 'Left-Kidney']


def abdomenmrct3d_4_palette():
    """AbdomenMRCT palette for external use."""
    return [[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
            [220, 248, 164]]


ixi2d_24_classes = oasis2d_24_classes
ixi2d_24_palette = oasis2d_24_palette
ixi3d_4_classes = oasis3d_4_classes
ixi3d_4_palette = oasis3d_4_palette
ixi3d_35_classes = oasis3d_35_classes
ixi3d_35_palette = oasis3d_35_palette

dataset_aliases = {
    'ixi_30': ['ixi', 'ixi_30'],
    'ixi2d_24': ['ixi2d', 'ixi2d_24'],
    'ixi3d_4': ['ixi3d_4'],
    'ixi3d_35': ['ixi3d', 'ixi3d_35'],
    'oasis2d_24': ['oasis2d', 'oasis2d_24'],
    'oasis3d_4': ['oasis3d_4'],
    'oasis3d_35': ['oasis3d', 'oasis3d_35'],
    'lpba40_54': ['lpba', 'lpba40', 'lpba40_54'],
    'abdomenmrct3d_4': ['abdomenmrct3d', 'abdomenmrct3d_4'],
}


def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_metainfo(dataset: str):
    name = dataset.split('_')[0]
    return dict(
        dataset_type=f'{name}_registration_dataset',
        task_name='registration',
        classes=get_classes(dataset),
        palette=get_palette(dataset))
