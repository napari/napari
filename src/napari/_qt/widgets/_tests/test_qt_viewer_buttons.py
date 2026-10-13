from unittest.mock import Mock

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QContextMenuEvent
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QPushButton

from napari._app_model._app import get_app_model
from napari._qt.dialogs.qt_modal import QtPopup
from napari._qt.widgets import qt_viewer_buttons as qt_viewer_buttons_
from napari._qt.widgets.qt_viewer_buttons import (
    QtLayerButtons,
    QtViewerButtons,
)
from napari.components.viewer_model import ViewerModel
from napari.utils.camera_orientations import (
    DepthAxisOrientation,
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)
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
    viewer_buttons = qt_viewer_buttons[1]
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
    assert viewer_buttons.grid_stride_box.value() == viewer.canvas.grid.stride
    assert viewer_buttons.grid_width_box
    assert viewer_buttons.grid_width_box.value() == viewer.canvas.grid.shape[1]
    assert viewer_buttons.grid_height_box
    assert (
        viewer_buttons.grid_height_box.value() == viewer.canvas.grid.shape[0]
    )
    assert viewer_buttons.grid_spacing_box
    assert (
        viewer_buttons.grid_spacing_box.value() == viewer.canvas.grid.spacing
    )

    # check that widget controls value changes update viewer grid values
    viewer_buttons.grid_stride_box.setValue(2)
    assert viewer_buttons.grid_stride_box.value() == viewer.canvas.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.canvas.grid.shape[1]
    viewer_buttons.grid_height_box.setValue(2)
    assert (
        viewer_buttons.grid_height_box.value() == viewer.canvas.grid.shape[0]
    )
    viewer_buttons.grid_spacing_box.setValue(0.5)
    assert (
        viewer_buttons.grid_spacing_box.value() == viewer.canvas.grid.spacing
    )

    # check viewer grid values changes update popup widget controls values
    viewer.canvas.grid.stride = 1
    viewer.canvas.grid.shape = (-1, -1)
    viewer.canvas.grid.spacing = 0.1
    # popup needs to be relaunched to get widget controls with the new values
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            widget.close()
    viewer_buttons.gridViewButton.customContextMenuRequested.emit(QPoint())
    assert viewer_buttons.grid_stride_box.value() == viewer.canvas.grid.stride
    viewer_buttons.grid_width_box.setValue(2)
    assert viewer_buttons.grid_width_box.value() == viewer.canvas.grid.shape[1]
    assert (
        viewer_buttons.grid_height_box.value() == viewer.canvas.grid.shape[0]
    )
    assert (
        viewer_buttons.grid_spacing_box.value() == viewer.canvas.grid.spacing
    )


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

    # check camera orientation combo changes affect viewer camera orientation
    assert viewer_buttons.vertical_combo
    assert (
        viewer_buttons.vertical_combo.currentEnum()
        == viewer.scene.camera.orientation[1]
    )
    assert viewer_buttons.horizontal_combo
    assert (
        viewer_buttons.horizontal_combo.currentEnum()
        == viewer.scene.camera.orientation[2]
    )
    assert viewer.scene.camera.handedness.value == 'right'

    viewer_buttons.vertical_combo.setCurrentEnum(VerticalAxisOrientation.UP)
    viewer_buttons.horizontal_combo.setCurrentEnum(
        HorizontalAxisOrientation.LEFT
    )
    assert viewer.scene.camera.orientation[1].value == 'up'
    assert viewer.scene.camera.orientation[2].value == 'left'
    assert viewer.scene.camera.orientation2d[0].value == 'up'
    assert viewer.scene.camera.orientation2d[1].value == 'left'

    # check zoom slider change affects viewer camera zoom
    assert viewer_buttons.zoom
    viewer_buttons.zoom.setValue(3)
    assert viewer.scene.camera.zoom == viewer_buttons.zoom.value() == 3

    # toggle ndisplay to be able to trigger ndisplay==3 popup
    viewer.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)

    # make ndisplay perspective setting popup
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    perspective_popup = None
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup

    # check camera orientation combo changes affect viewer camera orientation
    assert viewer_buttons.depth_combo
    assert (
        viewer_buttons.depth_combo.currentEnum()
        == viewer.scene.camera.orientation[0]
    )
    assert viewer_buttons.vertical_combo
    assert (
        viewer_buttons.vertical_combo.currentEnum()
        == viewer.scene.camera.orientation[1]
    )
    assert viewer_buttons.horizontal_combo
    assert (
        viewer_buttons.horizontal_combo.currentEnum()
        == viewer.scene.camera.orientation[2]
    )
    # check the values set by testing 2D orientation popup are inherited
    assert (
        viewer_buttons.vertical_combo.currentEnum()
        == VerticalAxisOrientation.UP
    )
    assert (
        viewer_buttons.horizontal_combo.currentEnum()
        == HorizontalAxisOrientation.LEFT
    )
    assert viewer.scene.camera.handedness.value == 'right'
    assert (
        viewer_buttons.orientation_help_symbol.objectName()
        == 'righthand_label'
    )
    assert 'right-handed' in viewer_buttons.orientation_help_symbol.toolTip()

    viewer_buttons.depth_combo.setCurrentEnum(DepthAxisOrientation.AWAY)
    viewer_buttons.vertical_combo.setCurrentEnum(VerticalAxisOrientation.DOWN)
    viewer_buttons.horizontal_combo.setCurrentEnum(
        HorizontalAxisOrientation.RIGHT
    )
    assert viewer.scene.camera.handedness.value == 'left'
    assert (
        viewer_buttons.orientation_help_symbol.objectName() == 'lefthand_label'
    )
    assert 'left-handed' in viewer_buttons.orientation_help_symbol.toolTip()
    assert viewer.scene.camera.orientation[0].value == 'away'
    assert viewer.scene.camera.orientation[1].value == 'down'
    assert viewer.scene.camera.orientation[2].value == 'right'

    # check perspective slider change affects viewer camera perspective
    assert viewer_buttons.perspective
    viewer_buttons.perspective.setValue(5)
    assert (
        viewer.scene.camera.perspective
        == viewer_buttons.perspective.value()
        == 5
    )

    # check zoom slider change affects viewer camera zoom
    assert viewer_buttons.zoom
    viewer_buttons.zoom.setValue(5)
    assert viewer.scene.camera.zoom == viewer_buttons.zoom.value() == 5

    # check viewer camera rotation value affects camera angles
    assert viewer_buttons.rz
    assert viewer_buttons.ry
    assert viewer_buttons.rx
    viewer_buttons.rz.setValue(90)
    viewer_buttons.ry.setValue(45)
    viewer_buttons.rx.setValue(0)
    assert (
        viewer.scene.camera.angles
        == (
            viewer_buttons.rz.value(),
            viewer_buttons.ry.value(),
            viewer_buttons.rx.value(),
        )
        == (90, 45, 0)
    )

    # popup needs to be relaunched to get widget controls with the new values
    perspective_popup.close()
    perspective_popup = None

    # check viewer camera perspective value affects perspective popup slider
    # initial value
    viewer.scene.camera.perspective = 10
    viewer.scene.camera.zoom = 2
    viewer.scene.camera.angles = (0, 0, 90)
    viewer_buttons.ndisplayButton.customContextMenuRequested.emit(QPoint())
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            perspective_popup = widget
    assert perspective_popup
    assert viewer_buttons.perspective
    assert (
        viewer.scene.camera.perspective
        == viewer_buttons.perspective.value()
        == 10
    )
    assert viewer_buttons.zoom
    assert viewer.scene.camera.zoom == viewer_buttons.zoom.value() == 2
    assert viewer_buttons.rz
    assert viewer_buttons.ry
    assert viewer_buttons.rx
    assert (
        viewer.scene.camera.angles
        == (
            viewer_buttons.rz.value(),
            viewer_buttons.ry.value(),
            viewer_buttons.rx.value(),
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
            (lambda: viewer, ViewerModel, 100),
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
        qt_viewer_buttons_,
        'action_manager',
        action_manager_mock,
    )
    modifiers = Qt.KeyboardModifier.AltModifier
    qtbot.mouseClick(
        viewer_buttons.transposeDimsButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )
    action_manager_mock.trigger.assert_called_with('napari:rotate_layers')

    trigger_mock = Mock()
    monkeypatch.setattr(
        'napari.utils.action_manager.ActionManager.trigger', trigger_mock
    )
    qtbot.mouseClick(
        viewer_buttons.transposeDimsButton, Qt.MouseButton.LeftButton
    )
    trigger_mock.assert_called_with('napari:transpose_axes')


@pytest.fixture
def qt_layer_buttons(qtbot):
    # create viewer model and buttons
    viewer = ViewerModel()
    layer_buttons = QtLayerButtons(viewer)
    qtbot.addWidget(layer_buttons)

    yield viewer, layer_buttons

    # close still open popup widgets
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QtPopup):
            widget.close()
    layer_buttons.close()


def test_layer_buttons_checked_on_selection(qt_layer_buttons):
    """Test that new points/shapes buttons are checked when a layer is selected."""
    import numpy as np

    viewer, layer_buttons = qt_layer_buttons

    # Initially no selection, buttons should not be checked
    assert layer_buttons.newPointsButton.property('mode') == 'new_points'
    assert layer_buttons.newShapesButton.property('mode') == 'new_shapes'
    assert layer_buttons.newPointsButton.property('creation_state') == 'none'
    assert layer_buttons.newShapesButton.property('creation_state') == 'none'
    assert layer_buttons.newLabelsButton.property('creation_state') == 'none'
    # no layers in the viewer, labels button is enabled
    assert layer_buttons.newLabelsButton.isEnabled()

    data_layer = viewer.add_image(np.zeros((10, 10)))
    data_layer2 = viewer.add_image(np.zeros((10, 10)))
    points_layer = viewer.add_points(ndim=2, size=10)

    # Selecting a single layer: buttons indicate single-layer inheritance
    viewer.layers.selection = [data_layer]
    assert layer_buttons.newPointsButton.property('creation_state') == 'full'
    assert layer_buttons.newShapesButton.property('creation_state') == 'full'
    assert layer_buttons.newLabelsButton.property('creation_state') == 'full'

    viewer.layers.selection = [points_layer]

    assert layer_buttons.newPointsButton.property('creation_state') == 'full'
    assert layer_buttons.newShapesButton.property('creation_state') == 'full'
    assert (
        layer_buttons.newLabelsButton.property('creation_state') == 'partial'
    )

    # Selecting multiple layers: buttons indicate multi-layer inheritance
    viewer.layers.selection = [data_layer, data_layer2]
    assert (
        layer_buttons.newPointsButton.property('creation_state') == 'partial'
    )
    assert (
        layer_buttons.newShapesButton.property('creation_state') == 'partial'
    )
    assert (
        layer_buttons.newLabelsButton.property('creation_state') == 'partial'
    )

    # Clearing selection: buttons return to default state
    viewer.layers.selection.clear()
    assert layer_buttons.newPointsButton.property('creation_state') == 'none'
    assert layer_buttons.newShapesButton.property('creation_state') == 'none'
    assert layer_buttons.newLabelsButton.property('creation_state') == 'none'
    # layers present but none selected, labels button is disabled
    assert not layer_buttons.newLabelsButton.isEnabled()


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
def test_new_layer_left_click(qt_layer_buttons, qtbot, kind):
    viewer, buttons = qt_layer_buttons
    affine = np.eye(5)
    affine[0, 1] = 0.5
    source = viewer.add_image(
        np.zeros((5, 10, 20, 30)),
        scale=(7, 4, 2, 1),
        translate=(11, 12, 13, 14),
        axis_labels=('t', 'z', 'y', 'x'),
        units=('s', 'um', 'um', 'um'),
        affine=affine,
    )
    qtbot.mouseClick(
        getattr(buttons, f'new{kind}Button'), Qt.MouseButton.LeftButton
    )
    layer = viewer.layers[-1]
    assert layer.ndim == 4
    np.testing.assert_array_equal(layer.scale, source.scale)
    np.testing.assert_array_equal(layer.translate, source.translate)
    np.testing.assert_array_equal(
        layer.affine.affine_matrix, source.affine.affine_matrix
    )
    assert layer.axis_labels == source.axis_labels
    assert layer.units == source.units
    if kind == 'Labels':
        assert layer.data.shape == (5, 10, 20, 30)


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
@pytest.mark.parametrize('ndim', [2, 3, 4])
def test_new_layer_dimension_popup(qt_layer_buttons, qtbot, kind, ndim):
    viewer, buttons = qt_layer_buttons
    viewer.add_image(
        np.zeros((5, 10, 20, 30)), axis_labels=('t', 'z', 'y', 'x')
    )
    button = getattr(buttons, f'new{kind}Button')
    buttons.show()
    qtbot.mouseClick(button, Qt.MouseButton.RightButton)
    assert len(viewer.layers) == 1
    # Offscreen Qt does not generate the native context-menu event.
    if button.findChild(QtPopup) is None:
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(1, 1))
        QApplication.sendEvent(button, event)
    popup = button.findChild(QtPopup)
    assert popup is not None
    assert popup.isVisible()
    assert popup.objectName() == 'QtModalPopup'
    assert popup.frame.objectName() == 'QtPopupFrame'
    selector = popup.findChild(QComboBox, 'new_layer_axes')
    assert [selector.itemText(index) for index in range(selector.count())] == [
        't, z, y, x (4D)',
        'z, y, x (3D)',
        'y, x (2D)',
    ]
    assert selector.currentData() == 4
    assert 'Right-click' in button.toolTip()
    tooltip = popup.findChild(QLabel, 'help_label').toolTip()
    for metadata in ('scale', 'translation', 'units', 'axis labels', 'affine'):
        assert metadata in tooltip
    assert 'stored order, not display order' in tooltip
    selector.setCurrentIndex(4 - ndim)
    note = popup.findChild(QLabel, 'new_layer_note').text()
    if ndim == 4:
        assert note == 'Match selected layer'
    else:
        assert note == f'Shared across: {", ".join(("t", "z")[: 4 - ndim])}'
    assert len(viewer.layers) == 1
    qtbot.mouseClick(
        popup.findChild(QPushButton, 'new_layer_create'),
        Qt.MouseButton.LeftButton,
    )
    assert viewer.layers[-1].ndim == ndim
    if kind == 'Labels':
        assert viewer.layers[-1].data.shape == (5, 10, 20, 30)[-ndim:]


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
def test_dimension_popup_reference_changes(qt_layer_buttons, qtbot, kind):
    viewer, buttons = qt_layer_buttons
    button = getattr(buttons, f'new{kind}Button')
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChild(QtPopup)
    selector = popup.findChild(QComboBox)
    assert selector.count() == 1
    assert selector.currentData() == 2
    popup.close()
    QApplication.sendPostedEvents(None, 0)
    source = viewer.add_image(np.zeros((5, 10, 20, 30)))
    other = viewer.add_image(np.zeros((20, 30)))
    viewer.layers.selection = {source, other}
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChildren(QtPopup)[-1]
    assert popup.findChild(QComboBox).count() == 3
    assert (
        popup.findChild(QLabel, 'new_layer_note').text()
        == 'Match selected extent'
    )
    popup.close()
    viewer.layers.selection.clear()
    if kind == 'Labels':
        assert not button.isEnabled()
    else:
        button.customContextMenuRequested.emit(QPoint())
        popup = button.findChildren(QtPopup)[-1]
        assert (
            popup.findChild(QLabel, 'new_layer_note').text() == 'Match viewer'
        )
        selector = popup.findChild(QComboBox)
        assert selector.currentData() == 4
        selector.setCurrentIndex(1)
        qtbot.mouseClick(
            popup.findChild(QPushButton, 'new_layer_create'),
            Qt.MouseButton.LeftButton,
        )
        assert viewer.layers[-1].ndim == 3
        np.testing.assert_array_equal(viewer.layers[-1].scale, (1, 1, 1))


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
def test_dimension_popup_coupled_transform(qt_layer_buttons, qtbot, kind):
    viewer, buttons = qt_layer_buttons
    affine = np.eye(5)
    affine[0, 1] = 0.5
    viewer.add_image(np.zeros((5, 10, 20, 30)), affine=affine)
    button = getattr(buttons, f'new{kind}Button')
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChild(QtPopup)
    selector = popup.findChild(QComboBox)
    model = selector.model()
    assert model.item(0).isEnabled()
    assert not model.item(1).isEnabled()
    assert 'coupled' in model.item(1).toolTip()
    # Both coupled axes are discarded for 2D, so this reduction is safe.
    assert model.item(2).isEnabled()
    qtbot.keyClick(selector, Qt.Key.Key_Down)
    assert selector.currentData() == 2
    popup.close()


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
@pytest.mark.parametrize('ndim', [2, 3, 4, 5, 6])
def test_dimension_popup_higher_dimensions(
    qt_layer_buttons, qtbot, kind, ndim
):
    viewer, buttons = qt_layer_buttons
    shape = (2, 3, 4, 5, 6, 7)
    axes = ('a', 'b', 'c', 'z', 'y', 'x')
    viewer.add_image(np.zeros(shape), axis_labels=axes)
    button = getattr(buttons, f'new{kind}Button')
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChild(QtPopup)
    selector = popup.findChild(QComboBox)
    assert [selector.itemData(index) for index in range(selector.count())] == [
        6,
        5,
        4,
        3,
        2,
    ]
    selector.setCurrentIndex(6 - ndim)
    assert selector.currentText() == f'{", ".join(axes[-ndim:])} ({ndim}D)'
    qtbot.mouseClick(
        popup.findChild(QPushButton, 'new_layer_create'),
        Qt.MouseButton.LeftButton,
    )
    assert viewer.layers[-1].ndim == ndim
    assert viewer.layers[-1].axis_labels == axes[-ndim:]
    if kind == 'Labels':
        assert viewer.layers[-1].data.shape == shape[-ndim:]


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
def test_dimension_popup_cancel(qt_layer_buttons, qtbot, kind):
    viewer, buttons = qt_layer_buttons
    viewer.add_image(np.zeros((5, 10, 20, 30)))
    button = getattr(buttons, f'new{kind}Button')
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChild(QtPopup)
    popup.findChild(QComboBox).setCurrentIndex(1)
    qtbot.keyClick(popup, Qt.Key.Key_Escape)
    assert len(viewer.layers) == 1


@pytest.mark.parametrize('kind', ['Points', 'Shapes', 'Labels'])
def test_dimension_popup_stored_axis_order(qt_layer_buttons, qtbot, kind):
    viewer, buttons = qt_layer_buttons
    viewer.add_image(
        np.zeros((5, 10, 20, 30)), axis_labels=('t', 'z', 'y', 'x')
    )
    viewer.dims.order = (2, 0, 3, 1)
    button = getattr(buttons, f'new{kind}Button')
    button.customContextMenuRequested.emit(QPoint())
    popup = button.findChild(QtPopup)
    selector = popup.findChild(QComboBox)
    assert selector.itemText(0) == 't, z, y, x (4D)'
    selector.setCurrentIndex(1)
    assert selector.currentText() == 'z, y, x (3D)'
    assert (
        popup.findChild(QLabel, 'new_layer_note').text() == 'Shared across: t'
    )
    qtbot.mouseClick(
        popup.findChild(QPushButton, 'new_layer_create'),
        Qt.MouseButton.LeftButton,
    )
    assert viewer.layers[-1].axis_labels == ('z', 'y', 'x')
    assert viewer.dims.order == (2, 0, 3, 1)
    if kind == 'Labels':
        assert viewer.layers[-1].data.shape == (10, 20, 30)
