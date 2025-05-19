from __future__ import annotations

from typing import Any

from app_model.types import Action, ToggleRule

from napari._qt.qt_main_window import Window


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
