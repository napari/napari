from __future__ import annotations

import inspect
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from app_model import Action
from app_model.types import SubmenuItem, ToggleRule
from magicgui.type_map._magicgui import MagicFactory
from magicgui.widgets import FunctionGui, Widget
from npe2 import io_utils, plugin_manager as pm
from npe2.manifest import contributions

from napari.errors.reader_errors import MultipleReaderError
from napari.utils.translations import trans

if TYPE_CHECKING:
    from npe2.manifest import PluginManifest
    from npe2.manifest.contributions import WriterContribution
    from npe2.plugin_manager import PluginName
    from npe2.types import LayerData, SampleDataCreator, WidgetCreator
    from qtpy.QtWidgets import QMenu, QWidget

    from napari._app_model.constants import MenuId
    from napari.layers import Layer
    from napari.types import SampleDict
    from napari.viewer import Viewer


class _FakeHookimpl:
    def __init__(self, name) -> None:
        self.plugin_name = name


def read(
    paths: Sequence[str], plugin: Optional[str] = None, *, stack: bool
) -> Optional[Tuple[List[LayerData], _FakeHookimpl]]:
    """Try to return data for `path`, from reader plugins using a manifest."""

    # do nothing if `plugin` is not an npe2 reader
    if plugin:
        # user might have passed 'plugin.reader_contrib' as the command
        # so ensure we check vs. just the actual plugin name
        plugin_name = plugin.partition('.')[0]
        if plugin_name not in get_readers():
            return None

    assert stack is not None
    # the goal here would be to make read_get_reader of npe2 aware of "stack",
    # and not have this conditional here.
    # this would also allow the npe2-npe1 shim to do this transform as well
    if stack:
        npe1_path = paths
    else:
        assert len(paths) == 1
        npe1_path = paths[0]
    try:
        layer_data, reader = io_utils.read_get_reader(
            npe1_path, plugin_name=plugin
        )
    except ValueError as e:
        # plugin wasn't passed and no reader was found
        if 'No readers returned data' not in str(e):
            raise
    else:
        return layer_data, _FakeHookimpl(reader.plugin_name)
    return None


def write_layers(
    path: str,
    layers: List[Layer],
    plugin_name: Optional[str] = None,
    writer: Optional[WriterContribution] = None,
) -> Tuple[List[str], str]:
    """
    Write layers to a file using an NPE2 plugin.

    Parameters
    ----------
    path : str
        The path (file, directory, url) to write.
    layers : list of Layers
        The layers to write.
    plugin_name : str, optional
        Name of the plugin to write data with. If None then all plugins
        corresponding to appropriate hook specification will be looped
        through to find the first one that can write the data.
    writer : WriterContribution, optional
        Writer contribution to use to write given layers, autodetect if None.

    Returns
    -------
    (written paths, writer name) as Tuple[List[str],str]

    written paths: List[str]
        Empty list when no plugin was found, otherwise a list of file paths,
        if any, that were written.
    writer name: str
        Name of the plugin selected to write the data.
    """
    layer_data = [layer.as_layer_data_tuple() for layer in layers]

    if writer is None:
        try:
            paths, writer = io_utils.write_get_writer(
                path=path, layer_data=layer_data, plugin_name=plugin_name
            )
        except ValueError:
            return [], ''
        else:
            return paths, writer.plugin_name

    n = sum(ltc.max() for ltc in writer.layer_type_constraints())
    args = (path, *layer_data[0][:2]) if n <= 1 else (path, layer_data)
    res = writer.exec(args=args)
    if isinstance(
        res, str
    ):  # pragma: no cover # it shouldn't be... bad plugin.
        return [res], writer.plugin_name
    return res or [], writer.plugin_name


def get_widget_contribution(
    plugin_name: str, widget_name: Optional[str] = None
) -> Optional[Tuple[WidgetCreator, str]]:
    widgets_seen = set()
    for contrib in pm.iter_widgets():
        if contrib.plugin_name == plugin_name:
            if not widget_name or contrib.display_name == widget_name:
                return contrib.get_callable(), contrib.display_name
            widgets_seen.add(contrib.display_name)
    if widget_name and widgets_seen:
        msg = trans._(
            'Plugin {plugin_name!r} does not provide a widget named {widget_name!r}. It does provide: {seen}',
            plugin_name=plugin_name,
            widget_name=widget_name,
            seen=widgets_seen,
            deferred=True,
        )
        raise KeyError(msg)
    return None


def populate_qmenu(menu: QMenu, menu_key: str):
    """Populate `menu` from a `menu_key` offering in the manifest."""
    # TODO: declare somewhere what menu_keys are valid.

    def _wrap(cmd_):
        def _wrapped(*args):
            cmd_.exec(args=args)

        return _wrapped

    for item in pm.iter_menu(menu_key):
        if isinstance(item, contributions.Submenu):
            subm_contrib = pm.get_submenu(item.submenu)
            subm = menu.addMenu(subm_contrib.label)
            assert subm is not None
            populate_qmenu(subm, subm_contrib.id)
        else:
            cmd = pm.get_command(item.command)
            action = menu.addAction(cmd.title)
            assert action is not None
            action.triggered.connect(_wrap(cmd))


def file_extensions_string_for_layers(
    layers: Sequence[Layer],
) -> Tuple[Optional[str], List[WriterContribution]]:
    """Create extensions string using npe2.

    When npe2 can be imported, returns an extension string and the list
    of corresponding writers. Otherwise returns (None,[]).

    The extension string is a ";;" delimeted string of entries. Each entry
    has a brief description of the file type and a list of extensions. For
    example:

        "Images (*.png *.jpg *.tif);;All Files (*.*)"

    The writers, when provided, are the
    `npe2.manifest.io.WriterContribution` objects. There is one writer per
    entry in the extension string.
    """

    layer_types = [layer._type_string for layer in layers]
    writers = list(pm.iter_compatible_writers(layer_types))

    def _items():
        """Lookup the command name and its supported extensions."""
        for writer in writers:
            name = pm.get_manifest(writer.command).display_name
            title = (
                f"{name} {writer.display_name}"
                if writer.display_name
                else name
            )
            yield title, writer.filename_extensions

    # extension strings are in the format:
    #   "<name> (*<ext1> *<ext2> *<ext3>);;+"

    def _fmt_exts(es):
        return " ".join(f"*{e}" for e in es if e) if es else "*.*"

    return (
        ";;".join(f"{name} ({_fmt_exts(exts)})" for name, exts in _items()),
        writers,
    )


def get_readers(path: Optional[str] = None) -> Dict[str, str]:
    """Get valid reader plugin_name:display_name mapping given path.

    Iterate through compatible readers for the given path and return
    dictionary of plugin_name to display_name for each reader. If
    path is not given, return all readers.

    Parameters
    ----------
    path : str
        path for which to find compatible readers

    Returns
    -------
    Dict[str, str]
        Dictionary of plugin_name to display name
    """

    if path:
        return {
            reader.plugin_name: pm.get_manifest(reader.command).display_name
            for reader in pm.iter_compatible_readers([path])
        }
    return {
        mf.name: mf.display_name
        for mf in pm.iter_manifests()
        if mf.contributions.readers
    }


def iter_manifests(
    disabled: Optional[bool] = None,
) -> Iterator[PluginManifest]:
    yield from pm.iter_manifests(disabled=disabled)


def widget_iterator() -> Iterator[Tuple[str, Tuple[str, Sequence[str]]]]:
    # eg ('dock', ('my_plugin', ('My widget', MyWidget)))
    wdgs: DefaultDict[str, List[str]] = defaultdict(list)
    for wdg_contrib in pm.iter_widgets():
        wdgs[wdg_contrib.plugin_name].append(wdg_contrib.display_name)
    return (('dock', x) for x in wdgs.items())


def sample_iterator() -> Iterator[Tuple[str, Dict[str, SampleDict]]]:
    return (
        (
            # use display_name for user facing display
            plugin_name,
            {
                c.key: {'data': c.open, 'display_name': c.display_name}
                for c in contribs
            },
        )
        for plugin_name, contribs in pm.iter_sample_data()
    )


def get_sample_data(
    plugin: str, sample: str
) -> Tuple[Optional[SampleDataCreator], List[Tuple[str, str]]]:
    """Get sample data opener from npe2.

    Parameters
    ----------
    plugin : str
        name of a plugin providing a sample
    sample : str
        name of the sample

    Returns
    -------
    tuple
        - first item is a data "opener": a callable that returns an iterable of
          layer data, or None, if none found.
        - second item is a list of available samples (plugin_name, sample_name)
          if no data opener is found.
    """
    avail: List[Tuple[str, str]] = []
    for plugin_name, contribs in pm.iter_sample_data():
        for contrib in contribs:
            if plugin_name == plugin and contrib.key == sample:
                return contrib.open, []
            avail.append((plugin_name, contrib.key))
    return None, avail


def index_npe1_adapters():
    """Tell npe2 to import and index any discovered npe1 plugins."""
    pm.index_npe1_adapters()


def on_plugin_enablement_change(enabled: Set[str], disabled: Set[str]):
    """Callback when any npe2 plugins are enabled or disabled.

    'Disabled' means the plugin remains installed, but it cannot be activated,
    and its contributions will not be indexed
    """
    from napari.settings import get_settings

    plugin_settings = get_settings().plugins
    to_disable = set(plugin_settings.disabled_plugins)
    to_disable.difference_update(enabled)
    to_disable.update(disabled)
    plugin_settings.disabled_plugins = to_disable

    for plugin_name in enabled:
        # technically, you can enable (i.e. "undisable") a plugin that isn't
        # currently registered/available.  So we check to make sure this is
        # actually a registered plugin.
        if plugin_name in pm.instance():
            _register_manifest_actions(pm.get_manifest(plugin_name))


def on_plugins_registered(manifests: Set[PluginManifest]):
    """Callback when any npe2 plugins are registered.

    'Registered' means that a manifest has been provided or discovered.
    """
    for mf in manifests:
        if not pm.is_disabled(mf.name):
            _register_manifest_actions(mf)


# TODO: This is a separate function from `_get_samples_submenu_actions` so it
# can be easily deleted once npe1 is no longer supported.
def _rebuild_npe1_samples_menu() -> None:
    """Register submenu and actions for all npe1 plugins, clearing all first."""
    from napari._app_model import get_app
    from napari._app_model.constants import MenuGroup, MenuId
    from napari._qt.qt_viewer import QtViewer
    from napari.plugins import menu_item_template, plugin_manager

    app = get_app()
    # Unregister all existing npe1 sample menu actions and submenus
    if unreg := plugin_manager._unreg_sample_submenus:
        unreg()
    if unreg := plugin_manager._unreg_sample_actions:
        unreg()

    sample_actions: List[Action] = []
    for plugin_name, samples in plugin_manager._sample_data.items():
        multiprovider = len(samples) > 1
        if multiprovider:
            submenu_id = f'napari/file/samples/{plugin_name}'
            submenu = [
                (
                    MenuId.FILE_SAMPLES,
                    SubmenuItem(
                        submenu=submenu_id, title=trans._(plugin_name)
                    ),
                ),
            ]
        else:
            submenu_id = MenuId.FILE_SAMPLES
            submenu = []

        for sample_name, sample_dict in samples.items():

            def _add_sample(
                qt_viewer: QtViewer,
                plugin=plugin_name,
                sample=sample_name,
            ):
                from napari._qt.dialogs.qt_reader_dialog import (
                    handle_gui_reading,
                )

                try:
                    qt_viewer.viewer.open_sample(plugin, sample)
                except MultipleReaderError as e:
                    handle_gui_reading(
                        e.paths,
                        qt_viewer,
                        stack=False,
                    )

            display_name = sample_dict['display_name'].replace("&", "&&")
            if multiprovider:
                title = display_name
            else:
                title = menu_item_template.format(plugin_name, display_name)

            action: Action = Action(
                id=f"{plugin_name}:{display_name}",
                title=title,
                menus=[{'id': submenu_id, 'group': MenuGroup.NAVIGATION}],
                callback=_add_sample,
            )
            sample_actions.append(action)

        unreg_sample_submenus = app.menus.append_menu_items(submenu)
        plugin_manager._unreg_sample_submenus = unreg_sample_submenus
        unreg_sample_actions = app.register_actions(sample_actions)
        plugin_manager._unreg_sample_actions = unreg_sample_actions


def _get_contrib_parent_menu(
    multiprovider: bool,
    parent_menu: MenuId,
    mf: PluginManifest,
    group: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """Get parent menu of plugin contribution (samples/widgets).

    If plugin provides multiple contributions, create a new submenu item."""
    submenu: List[Tuple[str, SubmenuItem]] = []
    if multiprovider:
        submenu_id = f'{parent_menu}/{mf.name}'
        submenu = [
            (
                parent_menu,
                SubmenuItem(
                    submenu=submenu_id,
                    title=trans._(mf.display_name),
                    group=group,
                ),
            ),
        ]
    else:
        submenu_id = parent_menu
    return submenu_id, submenu


def _get_samples_submenu_actions(
    mf: PluginManifest,
) -> Tuple[List[Any], List[Any]]:
    """Get sample data submenu and actions for a single npe2 plugin manifest."""
    from napari._app_model.constants import MenuGroup, MenuId
    from napari.plugins import menu_item_template

    if TYPE_CHECKING:
        from napari._qt.qt_viewer import QtViewer

    # If no sample data, return
    if not mf.contributions.sample_data:
        return [], []

    sample_data = mf.contributions.sample_data
    multiprovider = len(sample_data) > 1
    submenu_id, submenu = _get_contrib_parent_menu(
        multiprovider,
        MenuId.FILE_SAMPLES,
        mf,
    )

    sample_actions: List[Action] = []
    for sample in sample_data:

        def _add_sample(
            qt_viewer: QtViewer,
            plugin=mf.name,
            sample=sample.key,
        ):
            from napari._qt.dialogs.qt_reader_dialog import handle_gui_reading

            try:
                qt_viewer.viewer.open_sample(plugin, sample)
            except MultipleReaderError as e:
                handle_gui_reading(
                    e.paths,
                    qt_viewer,
                    stack=False,
                )

        if multiprovider:
            title = sample.display_name
        else:
            title = menu_item_template.format(
                mf.display_name, sample.display_name
            )
        # To display '&' instead of creating a shortcut
        title = title.replace("&", "&&")

        action: Action = Action(
            id=f'{mf.name}:{sample.key}',
            title=title,
            menus=[{'id': submenu_id, 'group': MenuGroup.NAVIGATION}],
            callback=_add_sample,
        )
        sample_actions.append(action)
    return submenu, sample_actions


def _get_widget_viewer_param(widget_callable, widget_name) -> str:
    """Get widget parameter name (if any) and check type."""
    from qtpy.QtWidgets import QWidget

    from napari.viewer import Viewer

    if inspect.isclass(widget_callable) and issubclass(
        widget_callable,
        (QWidget, Widget),
    ):
        widget_param = ""
        # Inspection can fail when adding to bundle as it thinks widget is
        # a builtin
        try:
            sig = inspect.signature(widget_callable.__init__)
        except ValueError:
            pass
        else:
            for param in sig.parameters.values():
                if param.name == 'napari_viewer':
                    widget_param = param.name
                    break
                if param.annotation in (
                    'napari.viewer.Viewer',
                    Viewer,
                ):
                    widget_param = param.name
                    break

    elif isinstance(widget_callable, MagicFactory) or inspect.isfunction(
        widget_callable
    ):
        widget_param = ""
    else:
        raise TypeError(
            trans._(
                "'{widget}' must be `QtWidgets.QWidget` or `magicgui.widgets.Widget` subclass, `MagicFactory` instance or function. Please raise an issue in napari GitHub with the plugin and widget you were trying to use.",
                deferred=True,
                widget=widget_name,
            )
        )
    return widget_param


def _get_widget_callback(
    napari_viewer: Viewer,
    mf: PluginManifest,
    widget_name: str,
    name: str,
) -> Optional[Tuple[Union[FunctionGui, QWidget, Widget], str]]:
    """Toggle if widget already built otherwise return widget.

    Returned widget will be added to main window by a processor.
    Note for magicgui type widget contributions, `Viewer` injection is done by
    `magicgui.register_type`.
    """
    from napari.plugins import _npe2

    window = napari_viewer.window
    if name in window._dock_widgets:
        dock_widget = window._dock_widgets[name]
        if dock_widget.isVisible():
            dock_widget.hide()
        else:
            dock_widget.show()
        return None

    # Get widget param name (if any) and check type
    if widget_contribution := _npe2.get_widget_contribution(
        mf.name,
        widget_name,
    ):
        widget_callable, _ = widget_contribution
        widget_param = _get_widget_viewer_param(widget_callable, widget_name)

        kwargs = {}
        if widget_param:
            kwargs[widget_param] = napari_viewer
        return widget_callable(**kwargs), name
    return None


def _get_widgets_submenu_actions(
    mf: PluginManifest,
) -> Tuple[List[Any], List[Any]]:
    """Get widget submenu and actions for a single npe2 plugin manifest."""
    from napari._app_model.constants import MenuGroup, MenuId
    from napari.plugins import menu_item_template

    if TYPE_CHECKING:
        from napari._qt.qt_main_window import Window

    # If no widgets, return
    if not mf.contributions.widgets:
        return [], []

    widgets = mf.contributions.widgets
    multiprovider = len(widgets) > 1
    submenu_id, submenu = _get_contrib_parent_menu(
        multiprovider,
        MenuId.MENUBAR_PLUGINS,
        mf,
        MenuGroup.PLUGIN_MULTI_SUBMENU,
    )

    widget_actions: List[Action] = []
    for widget in widgets:
        full_name = menu_item_template.format(
            mf.display_name,
            widget.display_name,
        )

        def _widget_callback(
            napari_viewer: Viewer,
            widget_name: str = widget.display_name,
            name: str = full_name,
        ) -> Optional[Tuple[Union[FunctionGui, QWidget, Widget], str]]:
            return _get_widget_callback(
                napari_viewer=napari_viewer,
                mf=mf,
                widget_name=widget_name,
                name=name,
            )

        def _get_current_dock_status(
            window: Window,
            name: str = full_name,
        ):
            if name in window._dock_widgets:
                return window._dock_widgets[name].isVisible()
            return False

        title = full_name
        if multiprovider:
            title = widget.display_name
        # To display '&' instead of creating a shortcut
        title = title.replace("&", "&&")

        widget_actions.append(
            Action(
                id=f'{mf.name}:{widget.display_name}',
                title=title,
                callback=_widget_callback,
                menus=[
                    {
                        'id': submenu_id,
                        'group': MenuGroup.PLUGIN_SINGLE_CONTRIBUTIONS,
                    }
                ],
                toggled=ToggleRule(get_current=_get_current_dock_status),
            )
        )
    return submenu, widget_actions


def _register_manifest_actions(mf: PluginManifest) -> None:
    """Gather and register actions from a manifest.

    This is called when a plugin is registered or enabled and it adds the
    plugin's menus and submenus to the app model registry.
    """
    from napari._app_model import get_app
    from napari._qt._qapp_model.qactions import _provide_window

    app = get_app()
    actions, submenus = _npe2_manifest_to_actions(mf)
    samples_submenu, sample_actions = _get_samples_submenu_actions(mf)
    widgets_submenu, widget_actions = _get_widgets_submenu_actions(mf)
    context = pm.get_context(cast('PluginName', mf.name))

    # Register and connect dispose callback to plugin deactivate ('unregistered') event
    actions = actions + sample_actions + widget_actions
    if actions:
        context.register_disposable(app.register_actions(actions))
    submenus = submenus + samples_submenu + widgets_submenu
    if submenus:
        context.register_disposable(app.menus.append_menu_items(submenus))

    # Register dispose functions that remove plugin widgets from widget dictionary
    # `window._dock_widgets`
    if window := _provide_window():

        class Event(NamedTuple):
            value: str

        for widget in mf.contributions.widgets or ():
            widget_event = Event(widget.display_name)

            def _remove_widget(event=widget_event):
                window._remove_dock_widget(event)

            context.register_disposable(_remove_widget)


def _npe2_manifest_to_actions(
    mf: PluginManifest,
) -> Tuple[List[Action], List[Tuple[str, SubmenuItem]]]:
    """Gather actions and submenus from a npe2 manifest, export app_model types."""
    from app_model.types import Action, MenuRule

    from napari._app_model.constants._menus import is_menu_contributable

    menu_cmds: DefaultDict[str, List[MenuRule]] = DefaultDict(list)
    submenus: List[Tuple[str, SubmenuItem]] = []
    for menu_id, items in mf.contributions.menus.items():
        if is_menu_contributable(menu_id):
            for item in items:
                if isinstance(item, contributions.MenuCommand):
                    rule = MenuRule(id=menu_id, **_when_group_order(item))
                    menu_cmds[item.command].append(rule)
                else:
                    subitem = _npe2_submenu_to_app_model(item)
                    submenus.append((menu_id, subitem))

    # Filter sample data commands (not URIs) as they are registered via
    # `_get_samples_submenu_actions`
    sample_data_ids = {
        contrib.command
        for contrib in mf.contributions.sample_data or ()
        if hasattr(contrib, 'command')
    }
    # Filter widgets as are registered in `_get_widgets_submenu_actions`
    widget_ids = {widget.command for widget in mf.contributions.widgets or ()}

    # We want to register all `Actions` so they appear in the command pallete
    actions: List[Action] = []
    for cmd in mf.contributions.commands or ():
        if cmd.id not in sample_data_ids | widget_ids:
            actions.append(
                Action(
                    id=cmd.id,
                    title=cmd.title,
                    category=cmd.category,
                    tooltip=cmd.short_title or cmd.title,
                    icon=cmd.icon,
                    enablement=cmd.enablement,
                    callback=cmd.python_name or '',
                    menus=menu_cmds.get(cmd.id),
                    keybindings=[],
                )
            )

    return actions, submenus


def _when_group_order(
    menu_item: contributions.MenuItem,
) -> dict:
    """Extract when/group/order from an npe2 Submenu or MenuCommand."""
    group, _, _order = (menu_item.group or '').partition("@")
    try:
        order: Optional[float] = float(_order)
    except ValueError:
        order = None
    return {'when': menu_item.when, 'group': group or None, 'order': order}


def _npe2_submenu_to_app_model(subm: contributions.Submenu) -> SubmenuItem:
    """Convert a npe2 submenu contribution to an app_model SubmenuItem."""
    contrib = pm.get_submenu(subm.submenu)
    return SubmenuItem(
        submenu=contrib.id,
        title=contrib.label,
        icon=contrib.icon,
        **_when_group_order(subm),
        # enablement= ??  npe2 doesn't have this, but app_model does
    )
