from napari._app_model._app import get_app
from napari._qt._qapp_model.qactions._window import ToggleDockWidgetAction
from napari._qt.qt_main_window import Window
from napari.components import ViewerModel


def test_dock_widget_toggler():
    """Tests `ToggleDockWidgetAction` toggling works."""
    viewer = ViewerModel()
    window = Window(viewer)
    action = ToggleDockWidgetAction(
        id='some.command.id',
        title='Toggle Dock Widget',
        dock_widget='dockConsole',
    )

    app = get_app()
    app.register_action(action)

    with app.injection_store.register(providers={Window: lambda: window}):
        assert window._qt_viewer.dockConsole.isVisible() is False
        app.commands.execute_command('some.command.id')
        assert window._qt_viewer.dockConsole.isVisible() is True
        app.commands.execute_command('some.command.id')
        assert window._qt_viewer.dockConsole.isVisible() is False
