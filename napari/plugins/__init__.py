import os
import sys
from inspect import isclass, signature
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

from magicgui import magicgui
from napari_plugin_engine import HookImplementation, PluginManager

from ..types import AugmentedFunction, AugmentedWidget
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
function_widgets: Dict[Tuple[str, str], Tuple[Callable, dict, dict]] = dict()


def register_dock_widget(
    args: Union[AugmentedWidget, List[AugmentedWidget]],
    hookimpl: HookImplementation,
):
    from qtpy.QtWidgets import QWidget

    plugin_name = hookimpl.plugin_name
    hook_name = '`napari_experimental_provide_dock_widget`'
    for arg in args if isinstance(args, list) else [args]:
        if isinstance(arg, tuple):
            if not arg:
                warn(
                    f'Plugin {plugin_name!r} provided an invalid tuple to '
                    f'{hook_name}.  Skipping'
                )
                continue
            _cls = arg[0]
            kwargs = arg[1] if len(arg) > 1 else {}
        else:
            _cls, kwargs = (arg, {})

        if not (isclass(_cls) and issubclass(_cls, QWidget)):
            warn(
                f'Plugin {plugin_name!r} provided an invalid '
                f'widget type to {hook_name}: {_cls!r}. Widget ignored.'
            )
            continue

        if not isinstance(kwargs, dict):
            warn(
                f'Plugin {plugin_name!r} provided invalid kwargs '
                f'to {hook_name} for class {_cls.__name__}. Widget ignored.'
            )
            continue

        # Get widget name
        name = str(kwargs.get('name', '')) or camel_to_spaces(_cls.__name__)

        key = (plugin_name, name)
        if key in dock_widgets:
            warn(
                "Plugin '{}' has already registered a dock widget '{}' "
                'which has now been overwritten'.format(*key)
            )
        dock_widgets[key] = (_cls, kwargs)


valid_magic_kwargs = set(signature(magicgui).parameters)


def register_function_widget(
    args: Union[AugmentedFunction, List[AugmentedFunction]],
    hookimpl: HookImplementation,
):
    plugin_name = hookimpl.plugin_name
    hook_name = '`napari_experimental_provide_function_widget`'
    for arg in args if isinstance(args, list) else [args]:
        if isinstance(arg, tuple):
            if not arg:
                warn(
                    f'Plugin {plugin_name!r} provided an invalid tuple to '
                    f'{hook_name}. Skipping'
                )
                continue
            func = arg[0]
            magic_kwargs = arg[1] if len(arg) > 1 else {}
            dock_kwargs = arg[2] if len(arg) > 2 else {}  # type: ignore
        else:
            func, magic_kwargs, dock_kwargs = (arg, {}, {})

        if not callable(func):
            msg = (
                f'Plugin {plugin_name!r} provided a non-callable type to '
                f'{hook_name}: {type(func)!r}. Function widget ignored.'
            )
            if isinstance(func, tuple):
                msg += (
                    " To provide multiple function widgets please use "
                    "a LIST of tuples"
                )
            warn(msg)
            continue

        if not isinstance(magic_kwargs, dict):
            warn(
                f'Plugin {plugin_name!r} provided invalid type '
                f'({type(magic_kwargs)}) for 2nd argument to {hook_name} '
                f'for function {func.__name__!r}. Widget ignored.'
            )
            continue

        if set(magic_kwargs) - valid_magic_kwargs:
            warn(
                f'Plugin {plugin_name!r} provided invalid magicgui kwargs '
                f'to {hook_name} for function {func.__name__!r}: '
                f'{set(magic_kwargs) - valid_magic_kwargs}. Widget ignored.'
            )
            continue

        if not isinstance(dock_kwargs, dict):
            warn(
                f'Plugin {plugin_name!r} provided invalid type '
                f'({type(dock_kwargs)}) for 3nd argument to {hook_name} '
                f'for function {func.__name__!r}. Widget ignored.'
            )
            continue

        # Get function name
        name = str(dock_kwargs.get('name', '')) or func.__name__.replace(
            '_', ' '
        )

        key = (plugin_name, name)
        if key in function_widgets:
            warn(
                "Plugin '{}' has already registered a function widget '{}' "
                'which has now been overwritten'.format(*key)
            )

        function_widgets[key] = (func, magic_kwargs, dock_kwargs)


plugin_manager.hook.napari_experimental_provide_dock_widget.call_historic(
    result_callback=register_dock_widget, with_impl=True
)


plugin_manager.hook.napari_experimental_provide_function_widget.call_historic(
    result_callback=register_function_widget, with_impl=True
)

#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'

__all__ = ["PluginManager", "plugin_manager", 'menu_item_template']
