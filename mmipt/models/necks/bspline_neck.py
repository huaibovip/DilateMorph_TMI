# Copyright (c) MMIPT. All rights reserved.
import torch
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn import functional as F

from mmipt.registry import MODELS
from .diff_neck import DiffeomorphicTransform


@MODELS.register_module()
class BSplineTransform(BaseModule):

    def __init__(
        self,
        # ndim,
        img_size,
        cps=(3, 3, 3),
        # DiffeomorphicTransform
        nsteps=7,
        # scale=1.0,
        svf=True,
        normalization=False,
        init_cfg=None,
    ):
        """Compute dense displacement field of Cubic B-spline FFD
        transformation model from input control point parameters.

        Args:
            ndim: (int) image dimension
            img_size: (int or tuple) size of the image
            cps: (int or tuple) control point spacing in number of intervals between pixel/voxel centres
            svf: (bool) stationary velocity field formulation if True
            init_cfg (dict or List[dict], optional): Initialization config dict

        Returns:
            flow (Sequence[Tensor]): velocity and displacement
                fields predicted by network.
        """

        super(BSplineTransform, self).__init__(init_cfg=init_cfg)
        # self.ndim = ndim
        self.svf = svf
        self.img_size = img_size  # param_ndim_setup(img_size, self.ndim)
        self.stride = cps  # param_ndim_setup(cps, self.ndim)
        self.kernels = self._set_kernel()
        self.padding = [(len(k) - 1) // 2 for k in self.kernels]

        if self.svf:
            self.dt = DiffeomorphicTransform(
                img_size=img_size,
                nsteps=nsteps,
                normalization=normalization,
                padding_mode='zeros',
                init_cfg=init_cfg,
            )

    def _set_kernel(self):
        kernels = list()
        for s in self.stride:
            # 1d cubic b-spline kernels
            kernels += [cubic_bspline1d(s)]
        return kernels

    def compute_flow(self, x):
        # separable 1d transposed convolution
        flow = x
        for i, (k, s, p) in \
            enumerate(zip(self.kernels, self.stride, self.padding)):
            k = k.to(dtype=x.dtype, device=x.device)
            flow = conv1d(
                flow,
                dim=i + 2,
                kernel=k,
                stride=s,
                padding=p,
                transpose=True,
            )

        #  crop the output to image size
        slicer = (slice(0, flow.shape[0]), slice(0, flow.shape[1])) \
                 + tuple(slice(s, s + self.img_size[i]) for i, s in enumerate(self.stride))
        flow = flow[slicer]
        return flow

    def forward(self, x, *args, **kwargs):
        """
        Returns:
            flow (Sequence[Tensor]): velocity and displacement
                fields predicted by network.
        """
        disp = velocity = self.compute_flow(x)
        if self.svf:
            disp = self.dt(velocity, *args, **kwargs)
        return velocity, disp


def cubic_bspline_value(x: float, derivative: int = 0) -> float:
    r"""Evaluate 1-dimensional cubic B-spline."""

    t = abs(x)
    # outside local support region
    if t >= 2:
        return 0
    # 0-th order derivative
    if derivative == 0:
        if t < 1:
            return 2 / 3 + (0.5 * t - 1) * t**2
        return -((t - 2)**3) / 6
    # 1st order derivative
    if derivative == 1:
        if t < 1:
            return (1.5 * t - 2.0) * x
        if x < 0:
            return 0.5 * (t - 2)**2
        return -0.5 * (t - 2)**2
    # 2nd oder derivative
    if derivative == 2:
        if t < 1:
            return 3 * t - 2
        return -t + 2


def cubic_bspline1d(stride,
                    derivative: int = 0,
                    dtype=None,
                    device=None) -> torch.Tensor:
    r"""Cubic B-spline kernel for specified control point spacing.
    Args:
        stride: Spacing between control points with respect to original (upsampled) image grid.
        derivative: Order of cubic B-spline derivative.
    Returns:
        Cubic B-spline convolution kernel.
    """

    if dtype is None:
        dtype = torch.float
    if not isinstance(stride, int):
        (stride, ) = stride
    kernel = torch.ones(4 * stride - 1, dtype=dtype)
    radius = kernel.shape[0] // 2
    for i in range(kernel.shape[0]):
        kernel[i] = cubic_bspline_value(
            (i - radius) / stride, derivative=derivative)
    if device is None:
        device = kernel.device
    return kernel.to(device)


def conv1d(data: Tensor,
           kernel: Tensor,
           dim: int = -1,
           stride: int = 1,
           dilation: int = 1,
           padding: int = 0,
           transpose: bool = False) -> Tensor:
    r"""Convolve data with 1-dimensional kernel along specified dimension."""

    result = data.type(kernel.dtype)  # (n, ndim, h, w, d)
    result = result.transpose(dim, -1)  # (n, ndim, ..., shape[dim])
    shape_ = result.size()
    # groups = numel(shape_[1:-1])  # (n, nidim * shape[not dim], shape[dim])
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    # 3*w*d, 1, kernel_size
    weight = kernel.expand(groups, 1, kernel.shape[-1])
    # n, 3*w*d, shape[dim]
    result = result.reshape(shape_[0], groups, shape_[-1])
    conv_fn = F.conv_transpose1d if transpose else F.conv1d
    result = conv_fn(
        result,
        weight,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    result = result.reshape(shape_[0:-1] + result.shape[-1:])
    result = result.transpose(-1, dim)
    return result


def param_ndim_setup(param, ndim):
    """
    Check dimensions of paramters and extend dimension if needed.

    Args:
        param: (int/float, tuple or list) check dimension match if tuple or list is given,
                expand to `dim` by repeating if a single integer/float number is given.
        ndim: (int) data/model dimension

    Returns:
        param: (tuple)
    """
    if isinstance(param, (int, float)):
        param = (param, ) * ndim
    elif isinstance(param, (tuple, list)):
        assert len(param) == ndim, \
            f"Dimension ({ndim}) mismatch with data"
        param = tuple(param)
    else:
        raise TypeError("Parameter type not int, tuple or list")
    return param
