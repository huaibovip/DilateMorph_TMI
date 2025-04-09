# Copyright (c) MMIPT. All rights reserved.
from typing import Any, Sequence, Union

import torch
from mmengine import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, xavier_init)
from mmengine.registry import Registry
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from torch import nn
from torch.nn import init

from mmipt.registry import MODELS


class Compose(BaseModule):

    def __init__(self, cfgs: Sequence[dict], init_cfg=None):
        super(Compose, self).__init__(init_cfg=init_cfg)
        assert isinstance(cfgs, Sequence), 'cfgs is not a sequence'
        self.transforms = ModuleList([MODELS.build(cfg) for cfg in cfgs])

    def forward(self, x, *args, **kwargs):
        for t in self.transforms:
            if hasattr(nn, t.__class__.__name__):
                x = t(x)
            else:
                x = t(x, *args, **kwargs)
        return x


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device('cpu')


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def generation_init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal. Default: 'normal'.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal. Default: 0.02.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


def build_module(module: Union[dict, nn.Module], builder: Registry, *args,
                 **kwargs) -> Any:
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if isinstance(module, dict):
        return builder.build(module, *args, **kwargs)
    elif isinstance(module, nn.Module):
        return module
    else:
        raise TypeError(
            f'Only support dict and nn.Module, but got {type(module)}.')


def xformers_is_enable(verbose: bool = False) -> bool:
    """Check whether xformers is installed.
    Args:
        verbose (bool): Whether to print the log.

    Returns:
        bool: Whether xformers is installed.
    """
    from mmipt.utils import try_import
    xformers = try_import('xformers')
    if xformers is None and verbose:
        print_log('Do not support Xformers.', 'current')
    return xformers is not None


def set_xformers(module: nn.Module, prefix: str = '') -> nn.Module:
    """Set xformers' efficient Attention for attention modules.

    Args:
        module (nn.Module): The module to set xformers.
        prefix (str): The prefix of the module name.

    Returns:
        nn.Module: The module with xformers' efficient Attention.
    """

    if not xformers_is_enable():
        print_log('Do not support Xformers. Please install Xformers first. '
                  'The program will run without Xformers.')
        return

    for n, m in module.named_children():
        if hasattr(m, 'set_use_memory_efficient_attention_xformers'):
            # set xformers for Diffusers' Cross Attention
            m.set_use_memory_efficient_attention_xformers(True)
            module_name = f'{prefix}.{n}' if prefix else n
            print_log(
                'Enable Xformers for HuggingFace Diffusers\' '
                f'module \'{module_name}\'.', 'current')
        else:
            set_xformers(m, prefix=n)

    return module
