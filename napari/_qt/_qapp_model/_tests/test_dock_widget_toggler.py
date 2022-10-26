from napari._app_model._app import get_app
from napari._qt._qapp_model.qactions._window import ToggleDockWidgetAction


# def test_dock_widget_toggler(make_napari_viewer):
#     """Tests `ToggleDockWidgetAction` toggling works."""
#     viewer = make_napari_viewer(show=True)
#     action = ToggleDockWidgetAction(
#         id='some.command.id',
#         title='Toggle Dock Widget',
#         dock_widget='dockConsole',
#     )

#     app = get_app()
#     app.register_action(action)

#     # with app.injection_store.register(providers={Window: lambda: viewer.window}):
#     assert viewer.window._qt_viewer.dockConsole.isVisible() is False
#     app.commands.execute_command('some.command.id')
#     assert viewer.window._qt_viewer.dockConsole.isVisible() is True
#     app.commands.execute_command('some.command.id')
#     assert viewer.window._qt_viewer.dockConsole.isVisible() is False
