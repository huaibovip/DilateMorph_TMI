# Copyright (c) MMIPT. All rights reserved.
from typing import Callable, List, Optional, Tuple, Union
import copy
import pickle
import random

from mmengine.dataset import force_full_init

from mmipt.datasets import BasicVolumeDataset
from mmipt.datasets.basic_volume_dataset import IMG_EXTENSIONS
from mmipt.registry import DATASETS


@DATASETS.register_module()
class OASISRegistrationDataset(BasicVolumeDataset):
    """OASISRegistrationDataset for open source projects in MMipt.

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
        dataset_type='oasis_registration_dataset',
        task_name='registration',
        classes=[
            'Unknown', 'Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex',
            'Left-Lateral-Ventricle', 'Left-Inf-Lat-Ventricle',
            'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
            'Left-Thalamus-Proper*', 'Left-Caudate', 'Left-Putamen', 'Left-Pallidum',
            '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus',
            'Left-Amygdala', 'Left-Accumbens', 'Left-VentralDC',
            'Left-Vessel', 'Left-Choroid-Plexus',
            'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex',
            'Right-Lateral-Ventricle', 'Right-Inf-Lat-Ventricle',
            'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
            'Right-Thalamus-Proper*', 'Right-Caudate', 'Right-Putamen',
            'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
            'Right-Accumbens', 'Right-VentralDC', 'Right-Vessel',
            'Right-Choroid-Plexus'
        ],
        palette=[[0, 0, 0], [245, 245, 245], [205, 62, 78], [120, 18, 134],
                 [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
                 [122, 186, 220], [236, 13, 176], [12, 48, 255], [204, 182, 142],
                 [42, 204, 164], [119, 159, 176], [220, 216, 20],
                 [103, 255, 255], [255, 165, 0], [165, 42, 42], [160, 32, 240],
                 [0, 200, 200], [245, 245, 245], [205, 62, 78], [120, 18, 134],
                 [196, 58, 250], [220, 248, 164], [230, 148, 34], [0, 118, 14],
                 [122, 186, 220], [236, 13, 176], [12, 48, 255],
                 [220, 216, 20], [103, 255, 255], [255, 165, 0], [165, 42, 42],
                 [160, 32, 240], [0, 200, 200]])
    # yapf: enbale

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(source='All'),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 # for folder
                 filename_tmpl: dict = dict(source='{}'),
                 search_key: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
                 recursive: bool = False,
                 **kwards):

        super().__init__(ann_file, metainfo, data_root, data_prefix, pipeline,
                         test_mode, filename_tmpl, search_key,
                         backend_args, data_suffix, recursive, **kwards)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """
        only for scan-to-scan mode
        """
        if self.test_mode:
            return super().get_data_info(idx=idx)

        def _read_serialize_data(id):
            start_addr = 0 if id == 0 else self.data_address[id - 1].item()
            end_addr = self.data_address[id].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            return pickle.loads(bytes)  # type: ignore

        # idx2 = np.random.randint(self.__len__(), size=1)
        idx_list = [i for i in range(self.__len__())]
        idx_list.remove(idx)
        random.shuffle(idx_list)
        idx2 = idx_list[0]
        if self.serialize_data:
            data_info = _read_serialize_data(idx)
            data_info2 = _read_serialize_data(idx2)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
            data_info2 = copy.deepcopy(self.data_list[idx2])

        data_info['target_path'] = data_info2[f'{self.search_key}_path']

        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
