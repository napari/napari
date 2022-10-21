from napari import Viewer
from napari._app_model._app import get_app
from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari.components import ViewerModel


def test_viewer_toggler():
    viewer = ViewerModel()
    action = ViewerToggleAction(
        id='some.command.id',
        title='Toggle Axis Visibility',
        viewer_attribute='axes',
        sub_attribute='visible',
    )
    app = get_app()
    app.register_action(action)

    with app.injection_store.register(providers={Viewer: lambda: viewer}):
        assert viewer.axes.visible is False
        app.commands.execute_command('some.command.id')
        assert viewer.axes.visible is True
        app.commands.execute_command('some.command.id')
        assert viewer.axes.visible is False
