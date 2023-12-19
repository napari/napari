"""Plugin related functions that require Qt.

Non-Qt plugin functions can be found in: `napari/plugins/_npe2.py`
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from app_model import Action
from app_model.types import SubmenuItem, ToggleRule
from magicgui.type_map._magicgui import MagicFactory
from magicgui.widgets import FunctionGui, Widget
from npe2 import plugin_manager as pm
from qtpy.QtWidgets import QWidget

from napari._app_model import get_app
from napari._app_model.constants import MenuGroup, MenuId
from napari._app_model.injection._providers import _provide_viewer
from napari._qt._qapp_model.injection._qproviders import _provide_window
from napari.plugins import menu_item_template
from napari.plugins._npe2 import (
    _get_contrib_parent_menu,
    get_widget_contribution,
)
from napari.utils.events import Event
from napari.utils.translations import trans
from napari.viewer import Viewer

if TYPE_CHECKING:
    from npe2.manifest import PluginManifest
    from npe2.plugin_manager import PluginName
    from npe2.types import WidgetCreator


def _get_widget_viewer_param(
    widget_callable: WidgetCreator, widget_name: str
) -> str:
    """Get widget parameter name for `viewer` (if any) and check type."""
    if inspect.isclass(widget_callable) and issubclass(
        widget_callable,
        (QWidget, Widget),
    ):
        widget_param = ""
        try:
            sig = inspect.signature(widget_callable.__init__)
        except ValueError:  # pragma: no cover
            # Inspection can fail when adding to bundled viewer as it thinks widget is
            # a builtin
            pass
        else:
            for param in sig.parameters.values():
                if param.name == 'napari_viewer' or param.annotation in (
                    'napari.viewer.Viewer',
                    Viewer,
                ):
                    widget_param = param.name
                    break

    # For magicgui type widget contributions, `Viewer` injection is done by
    # `magicgui.register_type`.
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


def _toggle_or_get_widget(
    plugin: str,
    widget_name: str,
    name: str,
) -> Optional[Tuple[Union[FunctionGui, QWidget, Widget], str]]:
    """Toggle if widget already built otherwise return widget.

    Returned widget will be added to main window by a processor.
    Note for magicgui type widget contributions, `Viewer` injection is done by
    `magicgui.register_type` instead of a provider via annnotation.
    """
    viewer = _provide_viewer()
    if viewer is None:
        raise RuntimeError(  # pragma: no cover
            trans._(
                "No current `Viewer` found. Note that widgets cannot be opened in headless mode.",
                deferred=True,
            )
        )

    window = viewer.window
    if window and (dock_widget := window._dock_widgets.get(name)):
        dock_widget.setVisible(not dock_widget.isVisible())
        return None

    # Get widget param name (if any) and check type
    widget_callable, _ = get_widget_contribution(plugin, widget_name)
    widget_param = _get_widget_viewer_param(widget_callable, widget_name)

    kwargs = {}
    if widget_param:
        kwargs[widget_param] = viewer
    return widget_callable(**kwargs), name


def _get_current_dock_status(name: str) -> bool:
    window = _provide_window()
    if window is None:
        raise RuntimeError(  # pragma: no cover
            trans._(
                "No current `Window` found. Note that widgets cannot be opened in headless mode.",
                deferred=True,
            )
        )
    if name in window._dock_widgets:
        return window._dock_widgets[name].isVisible()
    return False


def _build_widgets_submenu_actions(
    mf: PluginManifest,
) -> Tuple[List[Tuple[str, SubmenuItem]], List[Action]]:
    """Build widget submenu and actions for a single npe2 plugin manifest."""
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

        _widget_callback = partial(
            _toggle_or_get_widget,
            plugin=mf.name,
            widget_name=widget.display_name,
            name=full_name,
        )
        _get_current_dock_status_partial = partial(
            _get_current_dock_status,
            name=full_name,
        )

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
                toggled=ToggleRule(
                    get_current=_get_current_dock_status_partial
                ),
            )
        )
    return submenu, widget_actions


def _register_widget_actions(mf: PluginManifest) -> None:
    """Register widget actions and submenus from a manifest.

    This is called when a plugin is registered or enabled and it adds the
    plugin's widget actions and submenus to the app model registry.
    """
    app = get_app()
    widgets_submenu, widget_actions = _build_widgets_submenu_actions(mf)

    context = pm.get_context(cast('PluginName', mf.name))
    if widget_actions:
        context.register_disposable(app.register_actions(widget_actions))
    if widgets_submenu:
        context.register_disposable(
            app.menus.append_menu_items(widgets_submenu)
        )

    # Register dispose functions to remove plugin widgets from widget dictionary
    # `window._dock_widgets`
    if window := _provide_window():
        for widget in mf.contributions.widgets or ():
            widget_event = Event(type_name="", value=widget.display_name)

            def _remove_widget(event: Event = widget_event) -> None:
                window._remove_dock_widget(event)

            context.register_disposable(_remove_widget)
