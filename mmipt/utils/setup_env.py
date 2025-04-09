# Copyright (c) MMIPT. All rights reserved.
import datetime
import importlib
import warnings
from types import ModuleType
from typing import Optional

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmipt into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmipt default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmipt`, and all registries will build modules from
            mmipt's registry node.
            To understand more about the registry, please refer
            to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html
            Defaults to True.
    """  # noqa
    import mmipt.datasets  # noqa: F401,F403
    import mmipt.engine  # noqa: F401,F403
    import mmipt.evaluation  # noqa: F401,F403
    import mmipt.models  # noqa: F401,F403
    import mmipt.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
            or not DefaultScope.check_instance_created('mmipt')
        if never_created:
            DefaultScope.get_instance('mmipt', scope_name='mmipt')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmipt':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmipt", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmipt". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmipt-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmipt')


def try_import(name: str) -> Optional[ModuleType]:
    """Try to import a module.

    Args:
        name (str): Specifies what module to import in absolute or relative
            terms (e.g. either pkg.mod or ..mod).
    Returns:
        ModuleType or None: If importing successfully, returns the imported
        module, otherwise returns None.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None
