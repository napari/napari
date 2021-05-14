import importlib
import sys
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from warnings import warn

from napari_plugin_engine import HookImplementation
from napari_plugin_engine import PluginManager as PluginManager
from typing_extensions import TypedDict

from ..types import AugmentedWidget, LayerData, SampleDict, WidgetCallable
from ..utils._appdirs import user_site_packages
from ..utils.events import EmitterGroup, EventedSet
from ..utils.misc import camel_to_spaces, running_as_bundled_app
from ..utils.translations import trans
from . import _builtins, hook_specifications


class PluginHookOption(TypedDict):
    """Custom type specifying plugin and enabled state."""

    plugin: str
    enabled: bool


CallOrderDict = Dict[str, List[PluginHookOption]]


class NapariPluginManager(PluginManager):
    ENTRY_POINT = 'napari.plugin'

    def __init__(self):
        super().__init__('napari', discover_entry_point=self.ENTRY_POINT)

        self.events = EmitterGroup(
            source=self, registered=None, enabled=None, disabled=None
        )
        self._blocked: EventedSet[str] = EventedSet()

        def _on_blocked_change(event):
            # things that are "added to the blocked list" become disabled
            for item in event.added:
                self.events.disabled(value=item)
            # things that are "removed from the blocked list" become enabled
            for item in event.removed:
                self.events.enabled(value=item)

        self._blocked.events.changed.connect(_on_blocked_change)

        with self.discovery_blocked():
            self.add_hookspecs(hook_specifications)

        self._sample_data: Dict[str, Dict[str, SampleDict]] = dict()
        self._dock_widgets: Dict[
            str, Dict[str, Tuple[WidgetCallable, Dict[str, Any]]]
        ] = dict()
        self._function_widgets: Dict[
            str, Dict[str, Callable[..., Any]]
        ] = dict()

        if sys.platform.startswith('linux') and running_as_bundled_app():
            sys.path.append(user_site_packages())

    def _initialize(self):
        with self.discovery_blocked():
            self.register(_builtins, name='builtins')
            if importlib.util.find_spec("skimage") is not None:
                from . import _skimage_data

                self.register(_skimage_data, name='scikit-image')

    def register(
        self, namespace: Any, name: Optional[str] = None
    ) -> Optional[str]:
        name = super().register(namespace, name=name)
        if name:
            self.events.registered(value=name)
        return name

    def call_order(self, first_result_only=True) -> CallOrderDict:
        """Returns the call order from the plugin manager.

        Returns
        -------
        call_order : CallOrderDict
            mapping of hook_specification name, to a list of dicts with keys:
            {'plugin', 'enabled'}.  Plugins earlier in the dict are called
            sooner.
        """

        order = {}
        for spec_name, caller in self.hooks.items():
            # no need to save call order unless we only use first result
            if first_result_only and not caller.is_firstresult:
                continue
            impls = caller.get_hookimpls()
            # no need to save call order if there is only a single item
            if len(impls) > 1:
                order[spec_name] = [
                    {'plugin': impl.plugin_name, 'enabled': impl.enabled}
                    for impl in reversed(impls)
                ]
        return order

    def set_call_order(self, new_order: CallOrderDict):
        """Sets the plugin manager call order to match SETTINGS plugin values.

        Note: Run this after load_settings_plugin_defaults, which
        sets the default values in SETTINGS.

        Parameters
        ----------
        new_order : CallOrderDict
            mapping of hook_specification name, to a list of dicts with keys:
            {'plugin', 'enabled'}.  Plugins earlier in the dict are called
            sooner.
        """
        for spec_name, hook_caller in self.hooks.items():
            order = []
            for p in new_order.get(spec_name, []):
                try:
                    hook_caller._set_plugin_enabled(p['plugin'], p['enabled'])
                    order.append(p['plugin'])
                except KeyError:
                    pass
            if order:
                hook_caller.bring_to_front(order)

    # SAMPLE DATA ---------------------------

    def register_sample_data(
        self,
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
            warn_message = trans._(
                'Plugin {plugin_name!r} provided a non-dict object to {hook_name!r}: data ignored.',
                deferred=True,
                plugin_name=plugin_name,
                hook_name=hook_name,
            )
            warn(message=warn_message)
            return

        _data = {}
        for name, datum in list(data.items()):
            if isinstance(datum, dict):
                if 'data' not in datum or 'display_name' not in datum:
                    warn_message = trans._(
                        'In {hook_name!r}, plugin {plugin_name!r} provided an invalid dict object for key {name!r} that does not have required keys: "data" and "display_name". Ignoring',
                        deferred=True,
                        hook_name=hook_name,
                        plugin_name=plugin_name,
                        name=name,
                    )
                    warn(message=warn_message)
                    continue
            else:
                datum = {'data': datum, 'display_name': name}

            if not (
                callable(datum['data'])
                or isinstance(datum['data'], (str, Path))
            ):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided invalid data for key {name!r} in the dict returned by {hook_name!r}. (Must be str, callable, or dict), got ({dtype}).',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                    hook_name=hook_name,
                    dtype=type(datum["data"]),
                )
                warn(message=warn_message)
                continue
            _data[name] = datum

        if plugin_name not in self._sample_data:
            self._sample_data[plugin_name] = {}

        self._sample_data[plugin_name].update(_data)

    def available_samples(self) -> Tuple[Tuple[str, str], ...]:
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
        return tuple(
            (p, s) for p in self._sample_data for s in self._sample_data[p]
        )

    # FUNCTION & DOCK WIDGETS -----------------------

    def iter_widgets(self) -> Iterator[Tuple[str, Tuple[str, Dict]]]:
        from itertools import chain, repeat

        dock_widgets = zip(repeat("dock"), self._dock_widgets.items())
        func_widgets = zip(repeat("func"), self._function_widgets.items())
        yield from chain(dock_widgets, func_widgets)

    def register_dock_widget(
        self,
        args: Union[AugmentedWidget, List[AugmentedWidget]],
        hookimpl: HookImplementation,
    ):

        plugin_name = hookimpl.plugin_name
        hook_name = '`napari_experimental_provide_dock_widget`'
        for arg in args if isinstance(args, list) else [args]:
            if isinstance(arg, tuple):
                if not arg:
                    warn_message = trans._(
                        'Plugin {plugin_name!r} provided an invalid tuple to {hook_name}.  Skipping',
                        deferred=True,
                        plugin_name=plugin_name,
                        hook_name=hook_name,
                    )
                    warn(message=warn_message)
                    continue
                _cls = arg[0]
                kwargs = arg[1] if len(arg) > 1 else {}
            else:
                _cls, kwargs = (arg, {})

            if not callable(_cls):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided a non-callable object (widget) to {hook_name}: {_cls!r}. Widget ignored.',
                    deferred=True,
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                    _cls=_cls,
                )
                warn(message=warn_message)

                continue

            if not isinstance(kwargs, dict):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided invalid kwargs to {hook_name} for class {clsname}. Widget ignored.',
                    deferred=True,
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                    clsname=_cls.__name__,
                )
                warn(message=warn_message)
                continue

            # Get widget name
            name = str(kwargs.get('name', '')) or camel_to_spaces(
                _cls.__name__
            )

            if plugin_name not in self._dock_widgets:
                # tried defaultdict(dict) but got odd KeyErrors...
                self._dock_widgets[plugin_name] = {}
            elif name in self._dock_widgets[plugin_name]:
                warn_message = trans._(
                    'Plugin {plugin_name!r} has already registered a dock widget {name!r} which has now been overwritten',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                )
                warn(message=warn_message)

            self._dock_widgets[plugin_name][name] = (_cls, kwargs)

    def register_function_widget(
        self,
        args: Union[Callable, List[Callable]],
        hookimpl: HookImplementation,
    ):
        plugin_name = hookimpl.plugin_name
        hook_name = '`napari_experimental_provide_function`'
        for func in args if isinstance(args, list) else [args]:
            if not isinstance(func, FunctionType):
                warn_message = trans._(
                    'Plugin {plugin_name!r} provided a non-callable type to {hook_name}: {functype!r}. Function widget ignored.',
                    deferred=True,
                    functype=type(func),
                    plugin_name=plugin_name,
                    hook_name=hook_name,
                )

                if isinstance(func, tuple):
                    warn_message += trans._(
                        " To provide multiple function widgets please use a LIST of callables",
                        deferred=True,
                    )
                warn(message=warn_message)
                continue

            # Get function name
            name = func.__name__.replace('_', ' ')

            if plugin_name not in self._function_widgets:
                # tried defaultdict(dict) but got odd KeyErrors...
                self._function_widgets[plugin_name] = {}
            elif name in self._function_widgets[plugin_name]:
                warn_message = trans._(
                    'Plugin {plugin_name!r} has already registered a function widget {name!r} which has now been overwritten',
                    deferred=True,
                    plugin_name=plugin_name,
                    name=name,
                )
                warn(message=warn_message)

            self._function_widgets[plugin_name][name] = func

    def discover_sample_data(self):
        if self._sample_data:
            return
        self.hook.napari_provide_sample_data.call_historic(
            result_callback=partial(self.register_sample_data), with_impl=True
        )

    def discover_widgets(self):
        """Trigger discovery of dock_widgets plugins.

        As a "historic" hook, this should only need to be called once.
        (historic here means that even plugins that are discovered after this
        is called will be added.)
        """
        if self._dock_widgets:
            return
        self.hook.napari_experimental_provide_dock_widget.call_historic(
            partial(self.register_dock_widget), with_impl=True
        )
        self.hook.napari_experimental_provide_function.call_historic(
            partial(self.register_function_widget), with_impl=True
        )

    def get_widget(
        self, plugin_name: str, widget_name: Optional[str] = None
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
        plg_wdgs = self._dock_widgets.get(plugin_name)
        if not plg_wdgs:
            msg = trans._(
                'Plugin {plugin_name!r} does not provide any dock widgets',
                plugin_name=plugin_name,
                deferred=True,
            )
            raise KeyError(msg)

        if not widget_name:
            if len(plg_wdgs) > 1:
                msg = trans._(
                    'Plugin {plugin_name!r} provides more than 1 dock_widget. Must also provide "widget_name" from {avail}',
                    avail=set(plg_wdgs),
                    plugin_name=plugin_name,
                    deferred=True,
                )
                raise ValueError(msg)

            widget_name = list(plg_wdgs)[0]
        else:
            if widget_name not in plg_wdgs:
                msg = trans._(
                    'Plugin {plugin_name!r} does not provide a widget named {widget_name!r}',
                    plugin_name=plugin_name,
                    widget_name=widget_name,
                    deferred=True,
                )
                raise KeyError(msg)

        return plg_wdgs[widget_name]
