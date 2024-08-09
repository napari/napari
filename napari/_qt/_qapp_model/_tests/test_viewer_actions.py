import pytest

from napari._app_model import get_app
from napari._qt._qapp_model.qactions._viewer import (
    Q_VIEWER_ACTIONS,
    Q_VIEWER_NEW_DELETE_ACTIONS,
)


@pytest.mark.parametrize(
    'viewer_action', Q_VIEWER_ACTIONS + Q_VIEWER_NEW_DELETE_ACTIONS
)
def test_viewer_actions_execute_command(viewer_action, make_napari_viewer):
    """
    Test viewer actions associated with the viewer and layer buttons via app-model `execute_command`.

    Viewer buttons:
        * Toggle Console
        * Toggle 2D/3D view
        * Rotate
        * Transpose
        * Toggle grid mode
        * Reset view
    Layer controls:
        * New Points layer
        * New Shapes layer
        * New Labels layer
        * Delete selected layer(s)

    Note:
        This test is here only to ensure app-model action dispatch mechanism
        is working for these actions (which use the `_provide_viewer` provider).

        To check a set of functional tests related to these actions you can
        see:
            * https://github.com/napari/napari/blob/main/napari/components/_tests/test_viewer_model.py
            * https://github.com/napari/napari/blob/main/napari/_qt/_tests/test_qt_viewer.py
            * https://github.com/napari/napari/blob/main/napari/_qt/widgets/_tests/test_qt_viewer_buttons.py
    """
    app = get_app()
    make_napari_viewer()
    command_id = viewer_action.id
    app.commands.execute_command(command_id)
