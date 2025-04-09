# Copyright (c) MMIPT. All rights reserved.
import torch


def get_unknown_tensor(trimap, unknown_value=128 / 255):
    """Get 1-channel unknown area tensor from the 3 or 1-channel trimap tensor.

    Args:
        trimap (Tensor): Tensor with shape (N, 3, H, W) or (N, 1, H, W).
        unknown_value (float): Scalar value indicating unknown region in
            trimap.
            If trimap is pre-processed using `'rescale_to_zero_one'`, then
            0 for bg, 128/255 for unknown, 1 for fg,
            and unknown_value should set to 128 / 255.
            If trimap is pre-processed by
            :meth:`FormatTrimap(to_onehot=False)`, then
            0 for bg, 1 for unknown, 2 for fg
            and unknown_value should set to 1.
            If trimap is pre-processed by
            :meth:`FormatTrimap(to_onehot=True)`, then
            trimap is 3-channeled, and this value is not used.

    Returns:
        Tensor: Unknown area mask of shape (N, 1, H, W).
    """
    if trimap.shape[1] == 3:
        # The three channels correspond to (bg mask, unknown mask, fg mask)
        # respectively.
        weight = trimap[:, 1:2, :, :].float()
    # elif 'to_onehot' in meta[0]:
    # key 'to_onehot' is added by pipeline `FormatTrimap`
    # 0 for bg, 1 for unknown, 2 for fg
    # weight = trimap.eq(1).float()
    else:
        # trimap is simply processed by pipeline `RescaleToZeroOne`
        # 0 for bg, 128/255 for unknown, 1 for fg
        weight = trimap.eq(unknown_value).float()
    return weight


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize vector with it's lengths at the last dimension. If `vector` is
    two-dimension tensor, this function is same as L2 normalization.

    Args:
        vector (torch.Tensor): Vectors to be normalized.

    Returns:
        torch.Tensor: Vectors after normalization.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def to_channel_first(tensor: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tensor: 4D (B,H,W,C) or 5D (B,D,H,W,C)
    """
    index = (0, 3, 1, 2) if tensor.ndim == 4 else (0, 4, 1, 2, 3)
    return tensor.permute(index).contiguous()


def to_channel_last(tensor: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tensor: 4D (B,C,H,W) or 5D (B,C,D,H,W)
    """
    index = (0, 2, 3, 1) if tensor.ndim == 4 else (0, 2, 3, 4, 1)
    return tensor.permute(index).contiguous()
