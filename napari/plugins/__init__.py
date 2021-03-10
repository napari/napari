import sys
from inspect import signature
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

from magicgui import magicgui
from napari_plugin_engine import HookImplementation, PluginManager

from ..types import AugmentedWidget
from ..utils._appdirs import user_site_packages
from ..utils.misc import camel_to_spaces, running_as_bundled_app
from . import _builtins, hook_specifications

if sys.platform.startswith('linux') and running_as_bundled_app():
    sys.path.append(user_site_packages())


if TYPE_CHECKING:
    from magicgui.widgets import FunctionGui
    from qtpy.QtWidgets import QWidget


# the main plugin manager instance for the `napari` plugin namespace.
plugin_manager = PluginManager('napari', discover_entry_point='napari.plugin')
with plugin_manager.discovery_blocked():
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')

WidgetCallable = Callable[..., Union['FunctionGui', 'QWidget']]
dock_widgets: Dict[
    str, Dict[str, Tuple[WidgetCallable, Dict[str, Any]]]
] = dict()
function_widgets: Dict[str, Dict[str, Callable[..., Any]]] = dict()


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

        if not callable(_cls):
            warn(
                f'Plugin {plugin_name!r} provided a non-callable object '
                f'(widget) to {hook_name}: {_cls!r}. Widget ignored.'
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

        if plugin_name not in dock_widgets:
            # tried defaultdict(dict) but got odd KeyErrors...
            dock_widgets[plugin_name] = {}
        elif name in dock_widgets[plugin_name]:
            warn(
                "Plugin '{}' has already registered a dock widget '{}' "
                'which has now been overwritten'.format(plugin_name, name)
            )

        dock_widgets[plugin_name][name] = (_cls, kwargs)


def get_plugin_widget(
    plugin_name: str, widget_name: Optional[str] = None
) -> Tuple[WidgetCallable, Dict[str, Any]]:
    """Get widget `widget_name` provided by plugin `plugin_name`.

    Note: it's important that :func:`discover_dock_widgets` has been called
    first, otherwise plugins may not be found yet.  (Typically, that is done
    in qt_main_window)

    Parameters
    ----------
    plugin_name : str
        Name of a plugin providing a widget
    widget_name : str, optional
        Name of a widget provided by `plugin_name`. If `None`, and the
        specified plugin provides only a single widget, that widget will be
        returned, otherwise a ValueError will be raised, by default None

    Returns
    -------
    plugin_widget : Tuple[Callable, dict]
        Tuple of (widget_class, options).

    Raises
    ------
    KeyError
        If plugin `plugin_name` does not provide any widgets
    KeyError
        If plugin does not provide a widget named `widget_name`.
    ValueError
        If `widget_name` is not provided, but `plugin_name` provides more than
        one widget
    """
    plg_wdgs = dock_widgets.get(plugin_name)
    if not plg_wdgs:
        raise KeyError(
            f'Plugin {plugin_name!r} does not provide any dock widgets'
        )

    if not widget_name:
        if len(plg_wdgs) > 1:
            raise ValueError(
                f'Plugin {plugin_name!r} provides more than 1 dock_widget. '
                f'Must also provide "widget_name" from {set(plg_wdgs)}'
            )
        widget_name = list(plg_wdgs)[0]
    else:
        if widget_name not in plg_wdgs:
            raise KeyError(
                f'Plugin {plugin_name!r} does not provide '
                f'a widget named {widget_name!r}'
            )
    return plg_wdgs[widget_name]


_magicgui_sig = {
    name
    for name, p in signature(magicgui).parameters.items()
    if p.kind is p.KEYWORD_ONLY
}


def register_function_widget(
    args: Union[Callable, List[Callable]],
    hookimpl: HookImplementation,
):
    plugin_name = hookimpl.plugin_name
    hook_name = '`napari_experimental_provide_function`'
    for func in args if isinstance(args, list) else [args]:
        if not isinstance(func, FunctionType):
            msg = (
                f'Plugin {plugin_name!r} provided a non-callable type to '
                f'{hook_name}: {type(func)!r}. Function widget ignored.'
            )
            if isinstance(func, tuple):
                msg += (
                    " To provide multiple function widgets please use "
                    "a LIST of callables"
                )
            warn(msg)
            continue

        # Get function name
        name = func.__name__.replace('_', ' ')

        if plugin_name not in function_widgets:
            # tried defaultdict(dict) but got odd KeyErrors...
            function_widgets[plugin_name] = {}
        elif name in function_widgets[plugin_name]:
            warn(
                "Plugin '{}' has already registered a function widget '{}' "
                'which has now been overwritten'.format(plugin_name, name)
            )

        function_widgets[plugin_name][name] = func


def discover_dock_widgets():
    """Trigger discovery of dock_widgets plugins"""
    dw_hook = plugin_manager.hook.napari_experimental_provide_dock_widget
    dw_hook.call_historic(result_callback=register_dock_widget, with_impl=True)
    fw_hook = plugin_manager.hook.napari_experimental_provide_function
    fw_hook.call_historic(
        result_callback=register_function_widget, with_impl=True
    )


#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'

__all__ = ["PluginManager", "plugin_manager", 'menu_item_template']
