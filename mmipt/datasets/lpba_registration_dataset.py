# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, List, Optional, Tuple, Union

from mmipt.datasets import BasicVolumeDataset
from mmipt.datasets.basic_volume_dataset import IMG_EXTENSIONS
from mmipt.registry import DATASETS


@DATASETS.register_module()
class LPBARegistrationDataset(BasicVolumeDataset):
    """LPBARegistrationDataset for open source projects in MMipt.

    This dataset is designed for low-level vision tasks with medical image,
    such as registration.

    The annotation file is optional.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        filename_tmpl (dict): Template for each filename. Note that the
            template excludes the file extension. Default: dict().
        search_key (str): The key used for searching the folder to get
            data_list. Default: 'gt'.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        data_suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.
    """
    # yapf: disable
    METAINFO = dict(
        dataset_type='lpba_registration_dataset',
        task_name='registration',
        classes=[
            'Unknown', 'L-Superior-Frontal-Gyrus',
            'R-Superior-Frontal-Gyrus', 'L-Middle-Frontal-Gyrus',
            'R-Middle-Frontal-Gyrus', 'L-Inferior-Frontal-Gyrus',
            'R-Inferior-Frontal-Gyrus', 'L-Precentral-Gyrus',
            'R-Precentral-Gyrus', 'L-Middle-Orbitofrontal-Gyrus',
            'R-Middle-Orbitofrontal-Gyrus', 'L-Lateral-Orbitofrontal-Gyrus',
            'R-Lateral-Orbitofrontal-Gyrus', 'L-Gyrus-Rectus',
            'R-Gyrus-Rectus', 'L-Postcentral-Gyrus', 'R-Postcentral-Gyrus',
            'L-Superior-Parietal-Gyrus', 'R-Superior-Parietal-Gyrus',
            'L-Supramarginal-Gyrus', 'R-Supramarginal-Gyrus',
            'L-Angular-Gyrus', 'R-Angular-Gyrus', 'L-Precuneus', 'R-Precuneus',
            'L-Superior-Occipital-Gyrus', 'R-Superior-Occipital-Gyrus',
            'L-Middle-Occipital-Gyrus', 'R-Middle-Occipital-Gyrus',
            'L-Inferior-Occipital-Gyrus', 'R-Inferior-Occipital-Gyrus',
            'L-Cuneus', 'R-Cuneus', 'L-Superior-Temporal-Gyrus',
            'R-Superior-Temporal-Gyrus', 'L-Middle-Temporal-Gyrus',
            'R-Middle-Temporal-Gyrus', 'L-Inferior-Temporal-Gyrus',
            'R-Inferior-Temporal-Gyrus', 'L-Parahippocampal-Gyrus',
            'R-Parahippocampal-Gyrus', 'L-Lingual-Gyrus', 'R-Lingual-Gyrus',
            'L-Fusiform-Gyrus', 'R-Fusiform-Gyrus', 'L-Insular-Cortex',
            'R-Insular-Cortex', 'L-Cingulate-Gyrus', 'R-Cingulate-Gyrus',
            'L-Caudate', 'R-Caudate', 'L-Putamen', 'R-Putamen',
            'L-Hippocampus', 'R-Hippocampus' # 'Cerebellum', 'Brainstem'
        ],
        palette=[[0, 0, 0], [255, 128, 128], [255, 128, 0], [0, 0, 255],
                 [255, 128, 255], [128, 127, 0], [0, 0, 255], [0, 255, 255],
                 [128, 0, 0], [255, 0, 128], [128, 128, 64], [128, 0, 255],
                 [255, 128, 127], [128, 128, 255], [0, 255, 255], [0, 255, 0],
                 [128, 128, 64], [0, 127, 128], [0, 255, 255], [255, 255, 0],
                 [0, 128, 128], [255, 128, 255], [64, 0, 128], [128, 255, 128],
                 [0, 0, 255], [127, 127, 255], [255, 127, 0], [255, 127, 128],
                 [0, 127, 0], [160, 0, 0], [83, 166, 166], [255, 255, 127],
                 [0, 0, 128], [128, 0, 64], [0, 128, 255], [92, 92, 237],
                 [192, 128, 255], [17, 207, 255], [255, 0, 128],
                 [105, 180, 31], [64, 255, 0], [255, 255, 0], [0, 255, 255],
                 [35, 231, 216], [255, 0, 0], [0, 128, 255], [128, 255, 255],
                 [64, 128, 255], [255, 0, 255], [0, 48, 255], [255, 0, 128],
                 [128, 160, 48], [255, 128, 0], [133, 10, 31], [0, 255, 255]])

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(source='', target=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 filename_tmpl: dict = dict(source='S01', target='{}'),
                 search_key: Optional[str] = 'target',
                 backend_args: Optional[dict] = None,
                 data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
                 recursive: bool = False,
                 **kwards):

        super().__init__(ann_file, metainfo, data_root, data_prefix, pipeline,
                         test_mode, filename_tmpl, search_key, backend_args,
                         data_suffix, recursive, **kwards)
