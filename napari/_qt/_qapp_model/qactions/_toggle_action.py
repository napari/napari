from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app_model.types import Action, ToggleRule

from napari._qt.qt_main_window import Window

if TYPE_CHECKING:
    from napari.viewer import Viewer


class ViewerToggleAction(Action):
    """Action subclass that toggles a boolean viewer (sub)attribute on trigger.

    Parameters
    ----------
    id : str
        The command id of the action.
    title : str
        The title of the action. Prefer capital case.
    viewer_attribute : str
        The attribute of the viewer to toggle. (e.g. 'axes')
    sub_attribute : str
        The attribute of the viewer attribute to toggle. (e.g. 'visible')
    **kwargs
        Additional keyword arguments to pass to the Action constructor.

    Examples
    --------
    >>> action = ViewerToggleAction(
    ...     id='some.command.id',
    ...     title='Toggle Axis Visibility',
    ...     viewer_attribute='axes',
    ...     sub_attribute='visible',
    ... )
    """

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        title: str,
        viewer_attribute: str,
        sub_attribute: str,
        **kwargs: Any,
    ) -> None:
        def get_current(viewer: Viewer) -> bool:
            """return the current value of the viewer attribute"""
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, sub_attribute)

        def toggle(viewer: Viewer) -> None:
            """toggle the viewer attribute"""
            attr = getattr(viewer, viewer_attribute)
            setattr(attr, sub_attribute, not getattr(attr, sub_attribute))

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle,
            **kwargs,
        )


class DockWidgetToggleAction(Action):
    """`Action` subclass that toggles visibility of a `QtViewerDockWidget`.

    Parameters
    ----------
    id : str
        The command id of the action.
    title : str
        The title of the action. Prefer capital case.
    dock_widget: str
        The DockWidget to toggle.
    **kwargs
        Additional keyword arguments to pass to the `Action` constructor.

    Examples
    --------
    >>> action = DockWidgetToggleAction(
    ...     id='some.command.id',
    ...     title='Toggle Layer List',
    ...     dock_widget='dockConsole',
    ... )
    """

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        title: str,
        dock_widget: str,
        **kwargs: Any,
    ) -> None:
        def toggle_dock_widget(window: Window) -> None:
            dock_widget_prop = getattr(window._qt_viewer, dock_widget)
            dock_widget_prop.setVisible(not dock_widget_prop.isVisible())

        def get_current(window: Window) -> bool:
            dock_widget_prop = getattr(window._qt_viewer, dock_widget)
            return dock_widget_prop.isVisible()

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle_dock_widget,
            **kwargs,
        )
