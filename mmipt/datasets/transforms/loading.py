# Copyright (c) MMIPT. All rights reserved.
import os.path as osp
import pickle
from typing import List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import get_file_backend

from mmipt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load a single image or image frames from corresponding paths. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]
    - ori_[KEY]_shape
    - ori_[KEY]

    Args:
        key (str): Keys in results to find corresponding path.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            candidates are 'cv2', 'turbojpeg', 'pillow', and 'tifffile'.
            Defaults to None.
        use_cache (bool): If True, load all images at once. Default: False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        to_y_channel (bool): Whether to convert the loaded image to y channel.
            Only support 'rgb2ycbcr' and 'rgb2ycbcr'
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        key: str,
        color_type: str = 'color',
        channel_order: str = 'bgr',
        imdecode_backend: Optional[str] = None,
        use_cache: bool = False,
        to_float32: bool = False,
        to_y_channel: bool = False,
        save_original_img: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:

        self.key = key
        self.color_type = color_type
        self.channel_order = channel_order
        self.imdecode_backend = imdecode_backend
        self.save_original_img = save_original_img

        if backend_args is None:
            # lasy init at loading
            self.backend_args = None
            self.file_backend = None
        else:
            self.backend_args = backend_args.copy()
            self.file_backend = get_file_backend(backend_args=backend_args)

        # cache
        self.use_cache = use_cache
        self.cache = dict()

        # convert
        self.to_float32 = to_float32
        self.to_y_channel = to_y_channel

    def transform(self, results: dict) -> dict:
        """Functions to load image or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filenames = results[f'{self.key}_path']

        if not isinstance(filenames, (List, Tuple)):
            filenames = [str(filenames)]
            is_frames = False
        else:
            filenames = [str(v) for v in filenames]
            is_frames = True

        images = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []

        for filename in filenames:
            img = self._load_image(filename)
            img = self._convert(img)
            images.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())

        if not is_frames:
            images = images[0]
            shapes = shapes[0]
            if self.save_original_img:
                ori_imgs = ori_imgs[0]

        results[self.key] = images
        results[f'ori_{self.key}_shape'] = shapes
        results[f'{self.key}_channel_order'] = self.channel_order
        results[f'{self.key}_color_type'] = self.color_type
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results

    def _load_image(self, filename):
        """Load an image from file.

        Args:
            filename (str): Path of image file.
        Returns:
            np.ndarray: Image.
        """
        if self.file_backend is None:
            self.file_backend = get_file_backend(
                uri=filename, backend_args=self.backend_args)

        if (self.backend_args is not None) and (self.backend_args.get(
                'backend', None) == 'lmdb'):
            filename, _ = osp.splitext(osp.basename(filename))

        if filename in self.cache:
            img_bytes = self.cache[filename]
        else:
            img_bytes = self.file_backend.get(filename)
            if self.use_cache:
                self.cache[filename] = img_bytes

        img = mmcv.imfrombytes(
            content=img_bytes,
            flag=self.color_type,
            channel_order=self.channel_order,
            backend=self.imdecode_backend)

        return img

    def _convert(self, img: np.ndarray):
        """Convert an image to the require format.

        Args:
            img (np.ndarray): The original image.
        Returns:
            np.ndarray: The converted image.
        """

        if self.to_y_channel:

            if self.channel_order.lower() == 'rgb':
                img = mmcv.rgb2ycbcr(img, y_only=True)
            elif self.channel_order.lower() == 'bgr':
                img = mmcv.bgr2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        if self.to_float32:
            img = img.astype(np.float32)

        return img

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'key={self.key}, '
                    f'color_type={self.color_type}, '
                    f'channel_order={self.channel_order}, '
                    f'imdecode_backend={self.imdecode_backend}, '
                    f'use_cache={self.use_cache}, '
                    f'to_float32={self.to_float32}, '
                    f'to_y_channel={self.to_y_channel}, '
                    f'save_original_img={self.save_original_img}, '
                    f'backend_args={self.backend_args})')

        return repr_str


@TRANSFORMS.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load a pair of images from file.

    Each sample contains a pair of images, which are concatenated in the w
    dimension (a|b). This is a special loading class for generation paired
    dataset. It loads a pair of images as the common loader does and crops
    it into two images with the same shape in different domains.

    Required key is "pair_path". Added or modified keys are "pair",
    "pair_ori_shape", "ori_pair", "img_{domain_a}", "img_{domain_b}",
    "img_{domain_a}_path", "img_{domain_b}_path", "img_{domain_a}_ori_shape",
    "img_{domain_b}_ori_shape", "ori_img_{domain_a}" and
    "ori_img_{domain_b}".

    Args:
        key (str): Keys in results to find corresponding path.
        domain_a (str, Optional): One of the paired image domain. Defaults
            to 'A'.
        domain_b (str, Optional): The other of the paired image domain.
            Defaults to 'B'.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            candidates are 'cv2', 'turbojpeg', 'pillow', and 'tifffile'.
            Defaults to None.
        use_cache (bool): If True, load all images at once. Default: False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        to_y_channel (bool): Whether to convert the loaded image to y channel.
            Only support 'rgb2ycbcr' and 'rgb2ycbcr'
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        io_backend (str, optional): io backend where images are store. Defaults
            to None.
    """

    def __init__(self,
                 key: str,
                 domain_a: str = 'A',
                 domain_b: str = 'B',
                 color_type: str = 'color',
                 channel_order: str = 'bgr',
                 imdecode_backend: Optional[str] = None,
                 use_cache: bool = False,
                 to_float32: bool = False,
                 to_y_channel: bool = False,
                 save_original_img: bool = False,
                 backend_args: Optional[dict] = None):
        super().__init__(key, color_type, channel_order, imdecode_backend,
                         use_cache, to_float32, to_y_channel,
                         save_original_img, backend_args)
        assert isinstance(domain_a, str)
        assert isinstance(domain_b, str)
        self.domain_a = domain_a
        self.domain_b = domain_b

    def transform(self, results: dict) -> dict:
        """Functions to load paired images.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filename = results[f'{self.key}_path']

        image = self._load_image(filename)
        image = self._convert(image)
        if self.save_original_img:
            ori_image = image.copy()
        shape = image.shape

        # crop pair into a and b
        w = shape[1]
        if w % 2 != 0:
            raise ValueError(
                f'The width of image pair must be even number, but got {w}.')
        new_w = w // 2
        image_a = image[:, :new_w, :]
        image_b = image[:, new_w:, :]

        results[f'img_{self.domain_a}'] = image_a
        results[f'img_{self.domain_b}'] = image_b
        results[f'img_{self.domain_a}_path'] = filename
        results[f'img_{self.domain_b}_path'] = filename
        results[f'img_{self.domain_a}_ori_shape'] = image_a.shape
        results[f'img_{self.domain_b}_ori_shape'] = image_b.shape
        if self.save_original_img:
            results[f'ori_img_{self.domain_a}'] = image_a.copy()
            results[f'ori_img_{self.domain_b}'] = image_b.copy()

        results[self.key] = image
        results[f'ori_{self.key}_shape'] = shape
        results[f'{self.key}_channel_order'] = self.channel_order
        results[f'{self.key}_color_type'] = self.color_type
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_image

        return results


@TRANSFORMS.register_module()
class LoadVolumeFromFile(BaseTransform):
    """Load a single volume from corresponding path. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]
    - ori_[KEY]_shape

    Args:
        keys (Union[str, Sequence[str]]): Keys in results to find corresponding path.
        use_cache (bool): If True, load all volumes at once. Default: False.
        to_dtype (str): Convert the loaded img volume to a float32 numpy array.
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        use_cache: bool = False,
        to_dtype: Optional[np.dtype] = np.float32,
    ) -> None:

        if not isinstance(keys, Sequence):
            keys = [keys]

        self.keys = keys

        # cache
        self.use_cache = use_cache
        self.cache = dict()

        # convert
        self.to_dtype = to_dtype

    def transform(self, results: dict) -> dict:
        """Functions to load volume or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded volume and meta information.
        """
        for key in self.keys:
            volume = self._load_volume(results[f'{key}_path'])
            results[key] = self._convert(volume, self.to_dtype)
            results[f'ori_{key}_shape'] = volume.shape

        return results

    def _load_volume(self, path):
        """Load an volume from file.

        Args:
            path (str): Path of volume file.
        Returns:
            np.ndarray: Image.
        """
        filename, _ = osp.splitext(osp.basename(path))
        if filename in self.cache:
            volume = self.cache[filename]
        else:
            if path.endswith(
                ('.nii', '.nii.gz', '.mgz', '.NII', '.NII.GZ', '.MGZ')):
                import nibabel as nib
                img = nib.load(path)
                volume = np.squeeze(img.dataobj)
                # affine = img.affine
            elif path.endswith(('.pkl', '.PKL')):
                with open(path, 'rb') as f:
                    volume = pickle.load(f)
            elif path.endswith(('.npy', '.NPY')):
                volume = np.load(path)
            elif path.endswith(('.npz', '.NPZ')):
                npz = np.load(path)
                volume = next(iter(npz.values())) if len(
                    npz.keys()) == 1 else npz['vol']
            else:
                raise ValueError('unknown filetype for %s' % filename)

            if self.use_cache:
                self.cache[filename] = volume

        return volume

    def _convert(self, volume: np.ndarray, dtype):
        """Convert an volume to the require format.

        Args:
            volume (np.ndarray): The original volume.
        Returns:
            np.ndarray: The converted volume.
        """

        if volume.ndim == 2:
            # note: channel last for `image_to_tensor` func in PackInputs
            volume = np.expand_dims(volume, axis=-1)
        elif volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)

        return volume.astype(dtype)

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, '
                    f'use_cache={self.use_cache}, '
                    f'to_dtype={self.to_dtype})')

        return repr_str


@TRANSFORMS.register_module()
class LoadBundleVolumeFromFile(LoadVolumeFromFile):
    """Load bundled volumes from corresponding path. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]_img
    - [KEY]_seg

    if return_seg:
    - ori_[KEY]_img_shape
    - ori_[KEY]_seg_shape

    Args:
        keys (Sequence[str]): Keys in results to find corresponding path.
        return_seg (bool): If True, return image and label volumes. Default: False.
        use_cache (bool): If True, load all volumes at once. Default: False.
        img_dtype (str): Convert the loaded img volume to a float32 numpy array.
        seg_dtype (str): Convert the loaded seg volume to a int16 numpy array.
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        return_seg: bool = False,
        use_cache: bool = False,
        img_dtype: Optional[np.dtype] = np.float32,
        seg_dtype: Optional[np.dtype] = np.int16,
    ) -> None:

        self.keys = keys
        self.return_seg = return_seg

        # cache
        self.use_cache = use_cache
        self.cache = dict()

        # convert
        self.img_dtype = img_dtype
        self.seg_dtype = seg_dtype

    def transform(self, results: dict) -> dict:
        """Functions to load volume or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded volume and meta information.
        """

        for key in self.keys:
            (img, seg) = self._load_volumes(results[f'{key}_path'])
            results[f'{key}_img_path'] = results[f'{key}_path']

            results[f'{key}_img'] = self._convert(img, self.img_dtype)
            results[f'ori_{key}_img_shape'] = img.shape

            if self.return_seg:
                results[f'{key}_seg_path'] = results[f'{key}_path']
                results[f'{key}_seg'] = self._convert(seg, self.seg_dtype)
                results[f'ori_{key}_seg_shape'] = seg.shape

        return results

    def _load_volumes(self, path):
        """Load two volumes (img and seg) from file.

        Args:
            path (str): Path of volume file.
        Returns:
            np.ndarray: Image.
        """
        volumes = super()._load_volume(path)

        assert len(
            volumes
        ) == 2, 'The bundled volume should contain both `img` and `seg` volumes.'

        return volumes

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, '
                    f'return_seg={self.return_seg}, '
                    f'use_cache={self.use_cache}, '
                    f'img_dtype={self.img_dtype}, '
                    f'seg_dtype={self.seg_dtype})')

        return repr_str


@TRANSFORMS.register_module()
class LoadQuadVolumeFromFile(LoadVolumeFromFile):
    """Load bundled volumes from corresponding path. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]_img
    - [KEY]_seg

    if return_seg:
    - ori_[KEY]_img_shape
    - ori_[KEY]_seg_shape

    Args:
        keys (Sequence[str]): Keys in results to find corresponding path.
        return_seg (bool): If True, return image and label volumes. Default: False.
        use_cache (bool): If True, load all volumes at once. Default: False.
        img_dtype (str): Convert the loaded img volume to a float32 numpy array.
        seg_dtype (str): Convert the loaded seg volume to a int16 numpy array.
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        return_seg: bool = False,
        use_cache: bool = False,
        img_dtype: Optional[np.dtype] = np.float32,
        seg_dtype: Optional[np.dtype] = np.int16,
    ) -> None:

        self.keys = keys
        self.return_seg = return_seg
        assert self.keys[0] == 'source', 'keys order error'

        # cache
        self.use_cache = use_cache
        self.cache = dict()

        # convert
        self.img_dtype = img_dtype
        self.seg_dtype = seg_dtype

    def transform(self, results: dict) -> dict:
        """Functions to load volume or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded volume and meta information.
        """

        (src_img, dst_img, src_seg,
         dst_seg) = self._load_volume(results['data_path'])
        results[f'{self.keys[0]}_img_path'] = results['data_path']
        results[f'{self.keys[1]}_img_path'] = results['data_path']

        results[f'{self.keys[0]}_img'] = self._convert(src_img, self.img_dtype)
        results[f'{self.keys[1]}_img'] = self._convert(dst_img, self.img_dtype)
        results[f'ori_{self.keys[0]}_img_shape'] = src_img.shape
        results[f'ori_{self.keys[1]}_img_shape'] = dst_img.shape

        if self.return_seg:
            results[f'{self.keys[0]}_seg_path'] = results['data_path']
            results[f'{self.keys[1]}_seg_path'] = results['data_path']

            results[f'{self.keys[0]}_seg'] = self._convert(
                src_seg, self.seg_dtype)
            results[f'{self.keys[1]}_seg'] = self._convert(
                dst_seg, self.seg_dtype)
            results[f'ori_{self.keys[0]}_seg_shape'] = src_seg.shape
            results[f'ori_{self.keys[1]}_seg_shape'] = dst_seg.shape

        return results

    def _load_volumes(self, path):
        """Load four volumes (img and seg) from file.

        Args:
            path (str): Path of volume file.
        Returns:
            np.ndarray: Image.
        """
        volumes = super()._load_volume(path)

        assert len(
            volumes
        ) == 4, 'The bundled volume should contain both `img` and `seg` volumes.'

        return volumes

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, '
                    f'return_seg={self.return_seg}, '
                    f'use_cache={self.use_cache}, '
                    f'img_dtype={self.img_dtype}, '
                    f'seg_dtype={self.seg_dtype})')

        return repr_str


@TRANSFORMS.register_module()
class LoadVolumeFromHDF5(BaseTransform):
    """Load a single volume from corresponding path. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]
    - ori_[KEY]_shape

    Args:
        keys (Union[str, Sequence[str]]): Keys in results to find corresponding path.
        data_paths (Dict): dataset paths.
        to_dtype (str): Convert the loaded img volume to a float32 numpy array.
    """

    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        data_paths: dict,
        to_dtype: Optional[np.dtype] = np.float32,
    ) -> None:

        if not isinstance(keys, Sequence):
            keys = [keys]

        if not isinstance(data_paths, dict):
            raise ValueError('expect data_paths to be dict, but get {}'.format(
                type(data_paths)))

        self.keys = keys
        self.data_paths = data_paths
        self.files = None

        # convert
        self.to_dtype = to_dtype

    @staticmethod
    def load_file(paths: dict) -> dict:
        import h5py
        files = dict()
        for name, path in paths.items():
            files[name] = h5py.File(path, mode='r')
        return files

    def transform(self, results: dict) -> dict:
        """Functions to load volume or frames.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded volume and meta information.
        """
        # Do not load hdf5 file inside __init__
        if getattr(self, 'files', None) is None:
            self.files = self.load_file(self.data_paths)

        for key in self.keys:
            volume = self._load_volume(results[f'{key}_path'])
            results[key] = self._convert(volume, self.to_dtype)
            results[f'ori_{key}_shape'] = volume.shape

        return results

    def _load_volume(self, key: str):
        """Load an volume from file.

        Args:
            path (str): Path of volume file.
        Returns:
            np.ndarray: Image.
        """
        _index = key.index('/', 1)
        data_name = key[1:_index]
        data_path = key[_index + 1:]
        if data_name in self.files:
            return self.files[data_name][data_path]
        else:
            raise KeyError('dataset "{}" ({}) not found'.format(
                data_name, data_path))

    def _convert(self, volume: np.ndarray, dtype):
        """Convert an volume to the require format.

        Args:
            volume (np.ndarray): The original volume.
        Returns:
            np.ndarray: The converted volume.
        """

        if volume.ndim == 2:
            # note: channel last for `image_to_tensor` func in PackInputs
            volume = np.expand_dims(volume, axis=-1)
        elif volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)

        return volume.astype(dtype)

    def __repr__(self):

        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, '
                    f'data_paths={self.data_paths}, '
                    f'to_dtype={self.to_dtype})')

        return repr_str
