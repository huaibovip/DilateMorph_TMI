# Copyright (c) MMIPT. All rights reserved.
import numpy as np
import torch


def identity_map(sz, dtype=np.float32):
    """Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0:sz[0]]
    elif dim == 2:
        id = np.mgrid[0:sz[0], 0:sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    else:
        raise ValueError(
            'Only dimensions 1-3 are currently supported for the identity map')
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1. / (np.array(sz) - 1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d] * 2 - 1

    return torch.from_numpy(id.astype(np.float32))


def not_normalized_identity_map(sz):
    """Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0:sz[0]]
    elif dim == 2:
        id = np.mgrid[0:sz[0], 0:sz[1]]
    elif dim == 3:
        # id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    else:
        raise ValueError(
            'Only dimensions 1-3 are currently supported for the identity map')
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))


def gen_identity_map(img_sz, resize_factor=1., normalized=True):
    """given displacement field,  add displacement on grid field  todo  now
    keep for reproduce  this function will be disabled in the next release,
    replaced by spacing version."""
    ndim = len(img_sz)
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(ndim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(ndim)]
    if normalized:
        grid = identity_map(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid


def generate_grid(grid_size, grid_step=8, line_thickness=1):
    grid = np.zeros(grid_size)
    for i in range(0, grid.shape[0], grid_step):
        grid[i + line_thickness - 1] = 1
    for j in range(0, grid.shape[1], grid_step):
        grid[:, j + line_thickness - 1] = 1
    # for k in range(0, grid.shape[2], grid_step):
    #     grid[:, :, k + line_thickness - 1] = 1
    grid = grid[None, None, ...]
    grid = torch.from_numpy(grid)
    return grid


if __name__ == '__main__':
    grid = gen_identity_map(img_sz=(3, 4), normalized=True)
    print(grid)
