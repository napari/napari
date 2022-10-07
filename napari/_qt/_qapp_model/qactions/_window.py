"""Actions related to the 'Window' menu that require Qt."""

from typing import List

from app_model.types import Action, ToggleRule

from ...._app_model.constants import CommandId, MenuGroup, MenuId
from ....utils.translations import trans
from ...qt_main_window import Window


class ToggleDockWidgetAction(Action):
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
    >>> action = ToggleDockWidgetAction(
    ...     id='some.command.id',
    ...     title='Toggle Layer List',
    ...     dock_widget='dockConsole',
    ... )
    """

    def __init__(
        self,
        *,
        id: str,
        title: str,
        dock_widget: str,
        **kwargs,
    ):
        def toggle_dock_widget(window: Window):
            dock_widget_prop = getattr(window._qt_viewer, dock_widget)
            dock_widget_prop.setVisible(not dock_widget_prop.isVisible())

        def get_current(window: Window):
            dock_widget_prop = getattr(window._qt_viewer, dock_widget)
            return dock_widget_prop.isVisible()

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle_dock_widget,
            **kwargs,
        )


Q_WINDOW_ACTIONS: List[Action] = []

for cmd, dock_widget, status_tip in (
    (CommandId.TOGGLE_CONSOLE, 'dockConsole', 'Toggle console panel'),
    (
        CommandId.TOGGLE_LAYER_CONTROLS,
        'dockLayerControls',
        'Toggle layer controls panel',
    ),
    (CommandId.TOGGLE_LAYER_LIST, 'dockLayerList', 'Toggle layer list panel'),
):
    Q_WINDOW_ACTIONS.append(
        ToggleDockWidgetAction(
            id=cmd,
            title=cmd.title,
            dock_widget=dock_widget,
            menus=[
                {
                    'id': MenuId.MENUBAR_WINDOW,
                    'group': MenuGroup.NAVIGATION,
                }
            ],
            status_tip=trans._(status_tip),
        )
    )
