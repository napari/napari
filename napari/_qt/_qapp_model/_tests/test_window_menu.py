import pytest

from napari._app_model import get_app_model
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
    app = get_app_model()
    viewer = make_napari_viewer(show=True)
    widget = getattr(viewer.window._qt_viewer, action_dockwidget_name)
    widget_initial_visibility = widget.isVisible()
    action = viewer.window.window_menu.findAction(action_id)

    # ---- Check initial action checked state
    # Need to emit manually menu `aboutToShow` signal to ensure actions checked state is synced
    viewer.window.window_menu.aboutToShow.emit()
    # If the action is checked the widget should be visible
    # If the action is not checked the widget shouldn't be visible
    assert action.isChecked() == widget.isVisible()

    # ---- Check toggling dockwidget from initial visibility
    app.commands.execute_command(action_id)
    assert widget.isVisible() != widget_initial_visibility
    viewer.window.window_menu.aboutToShow.emit()
    assert action.isChecked() == widget.isVisible()

    # ---- Check restoring initial visibility
    app.commands.execute_command(action_id)
    assert widget.isVisible() == widget_initial_visibility
    viewer.window.window_menu.aboutToShow.emit()
    assert action.isChecked() == widget.isVisible()
