import os
import sys
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Type,
    Union,
)
from warnings import warn

from napari_plugin_engine import HookImplementation, PluginManager

from ..types import DockWidgetArg, MagicFunctionArg
from ..utils._appdirs import user_site_packages
from ..utils.misc import camel_to_spaces, running_as_bundled_app
from . import _builtins, hook_specifications

if sys.platform.startswith('linux') and running_as_bundled_app():
    sys.path.append(user_site_packages())


if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager(
    'napari', discover_entry_point='napari.plugin', discover_prefix='napari_'
)
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')


dock_widgets: Dict[Tuple[str, str], Tuple[Type['QWidget'], dict]] = dict()
functions: Dict[str, Type[Tuple[Callable, Dict, Dict]]] = dict()


def register_dock_widget(
    args: Union[DockWidgetArg, List[DockWidgetArg]],
    hookimpl: HookImplementation,
):
    from qtpy.QtWidgets import QWidget

    plugin_name = hookimpl.plugin_name
    for arg in args if isinstance(args, list) else [args]:
        if isinstance(arg, tuple):
            if not arg:
                warn(
                    f'Plugin {plugin_name} provided invalid tuple to '
                    '`register_dock_widget`.  Skipping'
                )
                continue
            _cls = arg[0]
            kwargs = arg[1] if len(arg) > 1 else {}
        else:
            _cls, kwargs = (arg, {})

        if not isclass(_cls) and issubclass(_cls, QWidget):
            warn(
                f'Plugin {plugin_name} provided invalid an invalid '
                f'widget type to `register_dock_widget`: {_cls!r}. '
                'Widget ignored.'
            )
            continue

        if not isinstance(kwargs, dict):
            warn(
                f'Plugin {plugin_name} provided invalid kwargs '
                f'to `register_dock_widget` for class {_cls.__name__}. '
                'Widget ignored.'
            )
            continue

        # Get widget name
        name = str(kwargs.get('name')) or camel_to_spaces(_cls.__name__)

        key = (plugin_name, name)
        if key in dock_widgets:
            warn(
                "Plugin '{}' has already registered a widget '{}' "
                'which has now been overwritten'.format(*key)
            )
        dock_widgets[key] = (_cls, kwargs)


def register_function(
    func: Union[MagicFunctionArg, List[MagicFunctionArg]], hookimpl
):
    for _func in func if isinstance(func, list) else [func]:
        if isinstance(_func, tuple):
            func_tuple = _func + ({},) * (3 - len(_func))
        else:
            func_tuple = (_func, {}, {})

        # Get function name
        name = func_tuple[2].get(
            'name', func_tuple[0].__name__.replace('_', ' ')
        )

        key = (hookimpl.plugin_name, name)
        if key in functions:
            warn(
                f'Plugin {key[0]} has already registered a function {key[1]} which has now been overwritten'
            )

        functions[(hookimpl.plugin_name, name)] = func_tuple


plugin_manager.hook.napari_experimental_provide_dock_widgets.call_historic(
    result_callback=register_dock_widget, with_impl=True
)


plugin_manager.hook.napari_experimental_provide_functions.call_historic(
    result_callback=register_function, with_impl=True
)

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'

__all__ = ["PluginManager", "plugin_manager", 'menu_item_template']
