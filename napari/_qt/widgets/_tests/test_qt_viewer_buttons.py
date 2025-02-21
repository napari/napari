from unittest.mock import Mock

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication

from napari._app_model._app import get_app_model
from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons
from napari.components.viewer_model import ViewerModel
from napari.viewer import Viewer


@pytest.fixture
def qt_viewer_buttons(qtbot):
    # create viewer model and buttons
    viewer = ViewerModel()
    viewer_buttons = QtViewerButtons(viewer)
    qtbot.addWidget(viewer_buttons)

    yield viewer, viewer_buttons

    # close still open popup widgets
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            widget.close()
    viewer_buttons.close()


def test_roll_dims_button_popup(qt_viewer_buttons, qtbot):
    """
    Make sure the QtViewerButtons.rollDimsButton popup works.
    """
    # get viewer model and buttons
    viewer, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.rollDimsButton

    # make dims order settings popup
    viewer_buttons.rollDimsButton.customContextMenuRequested.emit(QPoint())

    # check that the popup widget is available
    dims_sorter_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            dims_sorter_popup = widget
    assert dims_sorter_popup


def test_grid_view_button_popup(qt_viewer_buttons, qtbot):
    """
    Make sure the QtViewerButtons.gridViewbutton popup works.

    The popup widget should be able to show/change viewer grid settings.
    """
    # get viewer model and buttons
    viewer, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.gridViewButton

    # make grid settings popup
    viewer_buttons.gridViewButton.customContextMenuRequested.emit(QPoint())

    # check popup widgets were created
    assert viewer_buttons.grid_stride_box
    assert viewer_buttons.grid_stride_box.value() == viewer.grid.stride
    assert viewer_buttons.grid_width_box
    assert viewer_buttons.grid_width_box.value() == viewer.grid.shape[1]
    assert viewer_buttons.grid_height_box
    assert viewer_buttons.grid_height_box.value() == viewer.grid.shape[0]
    assert viewer_buttons.grid_spacing_box
    assert viewer_buttons.grid_spacing_box.value() == viewer.grid.spacing

    # check that widget controls value changes update viewer grid values
    viewer_buttons.grid_stride_box.setValue(2)
    assert viewer_buttons.grid_stride_box.value() == viewer.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.grid.shape[1]
    viewer_buttons.grid_height_box.setValue(2)
    assert viewer_buttons.grid_height_box.value() == viewer.grid.shape[0]
    viewer_buttons.grid_spacing_box.setValue(0.5)
    assert viewer_buttons.grid_spacing_box.value() == viewer.grid.spacing

    # check viewer grid values changes update popup widget controls values
    viewer.grid.stride = 1
    viewer.grid.shape = (-1, -1)
    viewer.grid.spacing = 0.1
    # popup needs to be relaunched to get widget controls with the new values
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            widget.close()
    viewer_buttons.gridViewButton.customContextMenuRequested.emit(QPoint())
    assert viewer_buttons.grid_stride_box.value() == viewer.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.grid.shape[1]
    assert viewer_buttons.grid_height_box.value() == viewer.grid.shape[0]
    assert viewer_buttons.grid_spacing_box.value() == viewer.grid.spacing


def test_ndisplay_button_popup(qt_viewer_buttons, qtbot):
    """
    Make sure the QtViewerButtons.ndisplayButton popup works.
    """
    # get viewer model and buttons
    viewer, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.ndisplayButton

    # trigger popup for ndisplay==2
    viewer.dims.ndisplay = 2
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    perspective_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup

    # check zoom slider change affects viewer camera zoom
    assert viewer_buttons.zoom
    viewer_buttons.zoom.setValue(3)
    assert viewer.camera.zoom == viewer_buttons.zoom.value() == 3

    # toggle ndisplay to be able to trigger ndisplay==3 popup
    viewer.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)

    # make ndisplay perspective setting popup
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    perspective_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup

    # check perspective slider change affects viewer camera perspective
    assert viewer_buttons.perspective
    viewer_buttons.perspective.setValue(5)
    assert viewer.camera.perspective == viewer_buttons.perspective.value() == 5

    # check zoom slider change affects viewer camera zoom
    assert viewer_buttons.zoom
    viewer_buttons.zoom.setValue(5)
    assert viewer.camera.zoom == viewer_buttons.zoom.value() == 5

    # check viewer camera rotation value affects camera angles
    assert viewer_buttons.rx
    assert viewer_buttons.ry
    assert viewer_buttons.rz
    viewer_buttons.rx.setValue(90)
    viewer_buttons.ry.setValue(45)
    viewer_buttons.rz.setValue(0)
    assert (
        viewer.camera.angles
        == (
            viewer_buttons.rx.value(),
            viewer_buttons.ry.value(),
            viewer_buttons.rz.value(),
        )
        == (90, 45, 0)
    )

    # popup needs to be relaunched to get widget controls with the new values
    perspective_popup.close()
    perspective_popup = None

    # check viewer camera perspective value affects perspective popup slider
    # initial value
    viewer.camera.perspective = 10
    viewer.camera.zoom = 2
    viewer.camera.angles = (0, 0, 90)
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup
    assert viewer_buttons.perspective
    assert (
        viewer.camera.perspective == viewer_buttons.perspective.value() == 10
    )
    assert viewer_buttons.zoom
    assert viewer.camera.zoom == viewer_buttons.zoom.value() == 2
    assert viewer_buttons.rx
    assert viewer_buttons.ry
    assert viewer_buttons.rz
    assert (
        viewer.camera.angles
        == (
            viewer_buttons.rx.value(),
            viewer_buttons.ry.value(),
            viewer_buttons.rz.value(),
        )
        == (0, 0, 90)
    )


def test_toggle_ndisplay(mock_app_model, qt_viewer_buttons, qtbot):
    """Check `toggle_ndisplay` works via `mouseClick`."""
    viewer, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.ndisplayButton

    app = get_app_model()

    assert viewer.dims.ndisplay == 2
    with app.injection_store.register(
        providers=[
            (lambda: viewer, Viewer, 100),
        ]
    ):
        qtbot.mouseClick(viewer_buttons.ndisplayButton, Qt.LeftButton)
        assert viewer.dims.ndisplay == 3


def test_transpose_rotate_button(monkeypatch, qt_viewer_buttons, qtbot):
    """
    Click should trigger `transpose_axes`. Alt/Option-click should trigger `rotate_layers.`
    """
    _, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.transposeDimsButton

    action_manager_mock = Mock(trigger=Mock())

    # Monkeypatch the action_manager instance to prevent viewer error
    monkeypatch.setattr(
        'napari._qt.widgets.qt_viewer_buttons.action_manager',
        action_manager_mock,
    )
    modifiers = Qt.AltModifier
    qtbot.mouseClick(
        viewer_buttons.transposeDimsButton, Qt.LeftButton, modifiers
    )
    action_manager_mock.trigger.assert_called_with('napari:rotate_layers')

    trigger_mock = Mock()
    monkeypatch.setattr(
        'napari.utils.action_manager.ActionManager.trigger', trigger_mock
    )
    qtbot.mouseClick(viewer_buttons.transposeDimsButton, Qt.LeftButton)
    trigger_mock.assert_called_with('napari:transpose_axes')
