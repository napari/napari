from napari import Viewer
from napari._app_model._app import get_app
from napari._qt._qapp_model.qactions._toggle_action import ViewerToggleAction
from napari.components import ViewerModel


def test_viewer_toggler(mock_app):
    viewer = ViewerModel()
    action = ViewerToggleAction(
        id='some.command.id',
        title='Toggle Axis Visibility',
        viewer_attribute='axes',
        sub_attribute='visible',
    )
    app = get_app()
    app.register_action(action)

    # Injection required as there is no current viewer, use a high weight (100)
    # so this provider is used over `_provide_viewer`, which would raise an error
    with app.injection_store.register(
        providers=[
            (lambda: viewer, Viewer, 100),
        ]
    ):
        assert viewer.axes.visible is False
        app.commands.execute_command('some.command.id')
        assert viewer.axes.visible is True
        app.commands.execute_command('some.command.id')
        assert viewer.axes.visible is False
