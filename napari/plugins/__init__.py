import importlib
import sys
from inspect import signature
from pathlib import Path
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

from magicgui import magicgui
from napari_plugin_engine import HookImplementation, PluginManager
from numpy import isin

from ..types import AugmentedWidget, LayerData, SampleDict
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
    if importlib.util.find_spec("skimage") is not None:
        from . import _skimage_data

        plugin_manager.register(_skimage_data, name='scikit-image')

WidgetCallable = Callable[..., Union['FunctionGui', 'QWidget']]
dock_widgets: Dict[
    str, Dict[str, Tuple[WidgetCallable, Dict[str, Any]]]
] = dict()
function_widgets: Dict[str, Dict[str, Callable[..., Any]]] = dict()
_sample_data: Dict[str, Dict[str, SampleDict]] = dict()


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


def register_sample_data(
    data: Dict[str, Union[str, Callable[..., Iterable[LayerData]]]],
    hookimpl: HookImplementation,
):
    """Register sample data dict returned by `napari_provide_sample_data`.

    Each key in `data` is a `sample_name` (the string that will appear in the
    `Open Sample` menu), and the value is either a string, or a callable that
    returns an iterable of ``LayerData`` tuples, where each tuple is a 1-, 2-,
    or 3-tuple of ``(data,)``, ``(data, meta)``, or ``(data, meta,
    layer_type)``.

    Parameters
    ----------
    data : Dict[str, Union[str, Callable[..., Iterable[LayerData]]]]
        A mapping of {sample_name->data}
    hookimpl : HookImplementation
        The hook implementation that returned the dict
    """
    plugin_name = hookimpl.plugin_name
    hook_name = 'napari_provide_sample_data'
    if not isinstance(data, dict):
        warn(
            f'Plugin {plugin_name!r} provided a non-dict object to '
            f'{hook_name!r}: data ignored.'
        )
        return

    _data = {}
    for name, datum in list(data.items()):
        if isinstance(datum, dict):
            if 'data' not in datum or 'display_name' not in datum:
                warn(
                    f'In {hook_name!r}, plugin {plugin_name!r} provided an '
                    f'invalid dict object for key {name!r} that does not have '
                    'required keys: "data" and "display_name".  Ignoring'
                )
                continue
        else:
            datum = {'data': datum, 'display_name': name}

        if not (
            callable(datum['data']) or isinstance(datum['data'], (str, Path))
        ):
            warn(
                f'Plugin {plugin_name!r} provided invalid data for key '
                f'{name!r} in the dict returned by {hook_name!r}. '
                f'(Must be str, callable, or dict), got ({type(datum["data"])}).'
            )
            continue
        _data[name] = datum

    if plugin_name not in _sample_data:
        _sample_data[plugin_name] = {}

    _sample_data[plugin_name].update(_data)


def discover_dock_widgets():
    """Trigger discovery of dock_widgets plugins"""
    dw_hook = plugin_manager.hook.napari_experimental_provide_dock_widget
    dw_hook.call_historic(result_callback=register_dock_widget, with_impl=True)
    fw_hook = plugin_manager.hook.napari_experimental_provide_function
    fw_hook.call_historic(
        result_callback=register_function_widget, with_impl=True
    )


def discover_sample_data():
    """Trigger discovery of sample data."""
    sd_hook = plugin_manager.hook.napari_provide_sample_data
    sd_hook.call_historic(result_callback=register_sample_data, with_impl=True)


def available_samples() -> Tuple[Tuple[str, str], ...]:
    """Return a tuple of sample data keys provided by plugins.

    Returns
    -------
    sample_keys : Tuple[Tuple[str, str], ...]
        A sequence of 2-tuples ``(plugin_name, sample_name)`` showing available
        sample data provided by plugins.  To load sample data into the viewer,
        use :meth:`napari.Viewer.open_sample`.

    Examples
    --------

    .. code-block:: python

        from napari.plugins import available_samples

        sample_keys = available_samples()
        if sample_keys:
            # load first available sample
            viewer.open_sample(*sample_keys[0])
    """
    return tuple((p, s) for p in _sample_data for s in _sample_data[p])


discover_sample_data()
def load_settings_plugin_defaults(SETTINGS):
    """Sets SETTINGS plugin defaults on start up from the defaults saved in
    the plugin manager.
    """
    plugins_call_order = []
    for name, hook_caller in plugin_manager.hooks.items():
        for hook_implementation in reversed(hook_caller._nonwrappers):
            plugins_call_order.append(
                (
                    name,
                    hook_implementation.plugin_name,
                    hook_implementation.enabled,
                )
            )

    SETTINGS._defaults['plugins'].plugins_call_order = plugins_call_order


def load_plugin_manager_settings(plugins_call_order):
    """Sets the plugin call order in the plugin in manger to match what is saved in
    SETTINGS.

    Note: Run this after load_settings_plugin_defaults, which sets the default values in
    SETTINGS.

    plugins_call_order : List of tuples
        [
            (spec_name, plugin_name, enabled),
            (spec_name, plugin_name, enabled),
        ]
    """
    # plugins_call_order = SETTINGS.plugins.plugins_call_order

    if plugins_call_order is not None:
        # (("get_write", "svg", True), ("get_writer", "builtins", True))
        for name, hook_caller in plugin_manager.hooks.items():
            # order: List of hook implementations
            order = []
            for spec_name, plugin_name, enabled in plugins_call_order:
                for hook_implementation in reversed(hook_caller._nonwrappers):
                    if (
                        spec_name == name
                        and hook_implementation.plugin_name == plugin_name
                    ):
                        hook_implementation.enabled = enabled
                        order.append(hook_implementation)
                if order:
                    hook_caller.bring_to_front(order)


#: Template to use for namespacing a plugin item in the menu bar
menu_item_template = '{}: {}'

__all__ = [
    "PluginManager",
    "plugin_manager",
    'menu_item_template',
    'dock_widgets',
    'function_widgets',
    'available_samples',
]
