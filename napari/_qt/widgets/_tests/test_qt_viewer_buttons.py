import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QApplication

from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons
from napari.components.viewer_model import ViewerModel


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

    # check that widget controls value changes update viewer grid values
    viewer_buttons.grid_stride_box.setValue(2)
    assert viewer_buttons.grid_stride_box.value() == viewer.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.grid.shape[1]
    viewer_buttons.grid_height_box.setValue(2)
    assert viewer_buttons.grid_height_box.value() == viewer.grid.shape[0]

    # check viewer grid values changes update popup widget controls values
    viewer.grid.stride = 1
    viewer.grid.shape = (-1, -1)
    # popup needs to be relaunched to get widget controls with the new values
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            widget.close()
    viewer_buttons.gridViewButton.customContextMenuRequested.emit(QPoint())
    assert viewer_buttons.grid_stride_box.value() == viewer.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.grid.shape[1]
    assert viewer_buttons.grid_height_box.value() == viewer.grid.shape[0]


def test_ndisplay_button_popup(qt_viewer_buttons, qtbot):
    """
    Make sure the QtViewerButtons.ndisplayButton popup works.
    """
    # get viewer model and buttons
    viewer, viewer_buttons = qt_viewer_buttons
    assert viewer_buttons.ndisplayButton

    # toggle ndisplay to be able to trigger popup
    viewer.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)

    # make ndisplay perspective setting popup
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    perspective_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup

    # check perspective slider change affects viewer camera perspective
    assert viewer_buttons.perspective_slider
    viewer_buttons.perspective_slider.setValue(5)
    assert (
        viewer.camera.perspective
        == viewer_buttons.perspective_slider.value()
        == 5
    )

    # popup needs to be relaunched to get widget controls with the new values
    perspective_popup.close()
    perspective_popup = None

    # check viewer camera perspective value affects perspective popup slider
    # initial value
    viewer.camera.perspective = 10
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup
    assert viewer_buttons.perspective_slider
    assert (
        viewer.camera.perspective
        == viewer_buttons.perspective_slider.value()
        == 10
    )
