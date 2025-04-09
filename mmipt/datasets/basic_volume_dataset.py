# Copyright (c) MMIPT. All rights reserved.
import os
import os.path as osp
import re
from typing import Callable, List, Optional, Tuple, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmipt.registry import DATASETS

IMG_EXTENSIONS = ('.mat', '.MAT', '.npy', '.NPY', '.pkl', '.PKL', 'nii', 'NII',
                  'nii.gz', 'NII.GZ', '.mgz', '.MGZ')


@DATASETS.register_module()
class BasicVolumeDataset(BaseDataset):
    """BasicVolumeDataset for open source projects in MMipt.

    This dataset is designed for low-level vision tasks with medical image,
    such as registration (atlas2scan, scan2atlas and fixed scan2scan)
    and segmentation tasks.
    
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

    def __init__(
            self,
            ann_file: str = '',
            metainfo: Optional[dict] = None,
            data_root: Optional[str] = None,
            data_prefix: dict = dict(),
            pipeline: List[Union[dict, Callable]] = [],
            test_mode: bool = False,
            # for folder
            filename_tmpl: dict = dict(),
            search_key: Optional[str] = None,
            backend_args: Optional[dict] = None,
            data_suffix: Optional[Union[str, Tuple[str]]] = IMG_EXTENSIONS,
            recursive: bool = False,
            **kwards):

        for key in data_prefix:
            if key not in filename_tmpl:
                filename_tmpl[key] = '{}'

        assert len(data_prefix) == len(
            filename_tmpl), 'len(data_prefix) != len(filename_tmpl)'

        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]

        self.search_key = search_key
        self.filename_tmpl = filename_tmpl
        self.use_ann_file = (ann_file != '')
        self.data_suffix = data_suffix
        self.recursive = recursive

        if backend_args is None:
            self.backend_args = None
        else:
            self.backend_args = backend_args.copy()
        self.file_backend = get_file_backend(
            uri=data_root, backend_args=backend_args)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwards)

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        data_list = []
        if self.use_ann_file:
            data_list = self._get_data_list_from_ann()
        else:
            data_list = self._get_data_list_from_folder()

        return data_list

    def _get_data_list_from_ann(self):
        """Get list of paths from annotation file.

        Returns:
            List: List of paths.
        """

        data_list = super().load_data_list()

        return data_list

    def _get_data_list_from_folder(self):
        """Get list of paths from folder.

        Returns:
            List: List of paths.
        """

        path_list = []
        folder = self.data_prefix[self.search_key]
        tmpl = self.filename_tmpl[self.search_key].format('')
        virtual_path = self.filename_tmpl[self.search_key].format('.*')
        for data_path in self.file_backend.list_dir_or_file(
                dir_path=folder,
                list_dir=False,
                suffix=self.data_suffix,
                recursive=self.recursive,
        ):
            basename, ext = osp.splitext(data_path)
            if re.match(virtual_path, basename):
                if basename.find(tmpl) == 0:
                    # prefix
                    data_path = data_path.replace(tmpl, '')
                else:
                    # suffix or others
                    data_path = data_path.replace(tmpl + ext, ext)
                path_list.append(data_path)

        # check data list TODO
        data_list = []
        for file in path_list:
            basename, ext = osp.splitext(file)
            if basename.startswith(os.sep):
                # Avoid absolute-path-like annotations
                basename = basename[1:]
            data = dict(key=basename)
            for key in self.data_prefix:
                path = osp.join(self.data_prefix[key],
                                (f'{self.filename_tmpl[key].format(basename)}'
                                 f'{ext}'))
                data[f'{key}_path'] = path
            data_list.append(data)

        return data_list
