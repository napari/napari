import pytest

from napari._app_model import get_app
from napari._qt._qapp_model.qactions._window import toggle_action_details
from napari._tests.utils import skip_local_popups


@skip_local_popups
@pytest.mark.parametrize(
    (
        'action_id',
        'action_text',
        'action_dockwidget_name',
        'action_status_tooltip',
    ),
    toggle_action_details,
)
def test_toggle_dockwidget_actions(
    make_napari_viewer,
    action_id,
    action_text,
    action_dockwidget_name,
    action_status_tooltip,
):
    app = get_app()
    viewer = make_napari_viewer(show=True)
    widget = getattr(viewer.window._qt_viewer, action_dockwidget_name)
    widget_initial_visibility = widget.isVisible()

    # Check toggling dockwidget from initial visibility
    app.commands.execute_command(action_id)
    assert widget.isVisible() != widget_initial_visibility

    # Check restoring initial visibility
    app.commands.execute_command(action_id)
    assert widget.isVisible() == widget_initial_visibility
