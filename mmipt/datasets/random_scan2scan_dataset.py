# Copyright (c) MMIPT. All rights reserved.
import copy
import pickle
import random

from mmengine.dataset import force_full_init

from mmipt.datasets import BasicVolumeDataset
from mmipt.registry import DATASETS


@DATASETS.register_module()
class RandomScan2ScanDataset(BasicVolumeDataset):
    """RandomScan2ScanDataset for open source projects in MMipt.

    This dataset is designed for low-level vision tasks with medical image,
    such as random scan-to-scan registration.

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

    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        assert self.search_key in ['source', 'source_img', 'source_seg']

    def __read_serialize_data(self, id):
        start_addr = 0 if id == 0 else self.data_address[id - 1].item()
        end_addr = self.data_address[id].item()
        bytes = memoryview(
            self.data_bytes[start_addr:end_addr])  # type: ignore
        return pickle.loads(bytes)  # type: ignore

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """
        only for scan-to-scan mode
        """
        if self.test_mode:
            return super().get_data_info(idx=idx)

        # idx2 = np.random.randint(self.__len__(), size=1)
        idx_list = [i for i in range(self.__len__())]
        idx_list.remove(idx)
        random.shuffle(idx_list)
        idx2 = idx_list[0]
        if self.serialize_data:
            data_info = self.__read_serialize_data(idx)
            data_info2 = self.__read_serialize_data(idx2)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
            data_info2 = copy.deepcopy(self.data_list[idx2])

        for src_key in self.data_prefix.keys():
            dst_key = src_key.replace('source', 'target')
            data_info[f'{dst_key}_path'] = data_info2[f'{src_key}_path']

        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
