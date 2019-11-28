import numpy as np

from napari.components import Dims
from napari._qt.qt_dims import QtDims
from qtpy.QtCore import Qt


def test_creating_view(qtbot):
    """
    Test creating dims view.
    """
    ndim = 4
    dims = Dims(ndim)
    view = QtDims(dims)

    qtbot.addWidget(view)

    # Check that the dims model has been appended to the dims view
    assert view.dims == dims

    # Check the number of displayed sliders is two less than the number of
    # dimensions
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2


def test_changing_ndim(qtbot):
    """
    Test changing the number of dimensions
    """
    ndim = 4
    view = QtDims(Dims(ndim))

    qtbot.addWidget(view)

    # Check that adding dimensions adds sliders
    view.dims.ndim = 5
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2

    # Check that removing dimensions removes sliders
    view.dims.ndim = 2
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2


def test_changing_focus(qtbot):
    """Test changing focus updates the dims.last_used prop."""
    # too-few dims, should have no sliders to update
    ndim = 2
    view = QtDims(Dims(ndim))
    assert view.last_used is None
    view.focus_down()
    view.focus_up()
    assert view.last_used is None

    view.dims.ndim = 5
    assert view.last_used == 2
    view.focus_down()
    assert view.last_used == 1
    view.focus_up()
    assert view.last_used == 2
    view.focus_up()
    assert view.last_used == 0
    view.focus_down()
    assert view.last_used == 2


def test_changing_display(qtbot):
    """
    Test changing the displayed property of an axis
    """
    ndim = 4
    view = QtDims(Dims(ndim))

    qtbot.addWidget(view)

    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2

    # Check changing displayed removes a slider
    view.dims.ndisplay = 3
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 3


def test_slider_values(qtbot):
    """
    Test the values of a slider stays matched to the values of the dims point.
    """
    ndim = 4
    view = QtDims(Dims(ndim))

    qtbot.addWidget(view)

    # Check that values of the dimension slider matches the values of the
    # dims point at initialization
    first_slider = view.slider_widgets[0].slider
    assert first_slider.value() == view.dims.point[0]

    # Check that values of the dimension slider matches the values of the
    # dims point after the point has been moved within the dims
    view.dims.set_point(0, 2)
    assert first_slider.value() == view.dims.point[0]

    # Check that values of the dimension slider matches the values of the
    # dims point after the point has been moved within the slider
    first_slider.setValue(1)
    assert first_slider.value() == view.dims.point[0]


def test_slider_range(qtbot):
    """
    Tests range of the slider is matched to the range of the dims
    """
    ndim = 4
    view = QtDims(Dims(ndim))

    qtbot.addWidget(view)

    # Check the range of slider matches the values of the range of the dims
    # at initialization
    first_slider = view.slider_widgets[0].slider
    assert first_slider.minimum() == view.dims.range[0][0]
    assert (
        first_slider.maximum() == view.dims.range[0][1] - view.dims.range[0][2]
    )
    assert first_slider.singleStep() == view.dims.range[0][2]

    # Check the range of slider stays matched to the values of the range of
    # the dims
    view.dims.set_range(0, (1, 5, 2))
    assert first_slider.minimum() == view.dims.range[0][0]
    assert (
        first_slider.maximum() == view.dims.range[0][1] - view.dims.range[0][2]
    )
    assert first_slider.singleStep() == view.dims.range[0][2]


def test_order_when_changing_ndim(qtbot):
    """
    Test order of the sliders when changing the number of dimensions.
    """
    ndim = 4
    view = QtDims(Dims(ndim))

    qtbot.addWidget(view)

    # Check that values of the dimension slider matches the values of the
    # dims point after the point has been moved within the dims
    view.dims.set_point(0, 2)
    view.dims.set_point(1, 1)
    for i in range(view.dims.ndim - 2):
        slider = view.slider_widgets[i].slider
        assert slider.value() == view.dims.point[i]

    # Check the matching dimensions and sliders are preserved when
    # dimensions are added
    view.dims.ndim = 5
    for i in range(view.dims.ndim - 2):
        slider = view.slider_widgets[i].slider
        assert slider.value() == view.dims.point[i]

    # Check the matching dimensions and sliders are preserved when dims
    # dimensions are removed
    view.dims.ndim = 4
    for i in range(view.dims.ndim - 2):
        slider = view.slider_widgets[i].slider
        assert slider.value() == view.dims.point[i]

    # Check the matching dimensions and sliders are preserved when dims
    # dimensions are removed
    view.dims.ndim = 3
    for i in range(view.dims.ndim - 2):
        slider = view.slider_widgets[i].slider
        assert slider.value() == view.dims.point[i]


def test_update_dims_labels(qtbot):
    """
    Test that the slider_widget axis labels are updated with the dims model
    and vice versa.
    """
    ndim = 4
    view = QtDims(Dims(ndim))
    qtbot.addWidget(view)
    view.dims.axis_labels = list('TZYX')
    assert [w.label.text() for w in view.slider_widgets] == list('TZYX')

    first_label = view.slider_widgets[0].label
    assert first_label.text() == view.dims.axis_labels[0]
    first_label.setText('napari')
    first_label.editingFinished.emit()
    assert first_label.text() == view.dims.axis_labels[0]


def test_slider_press_updates_last_used(qtbot):
    """pressing on the slider should update the dims.last_used property"""
    ndim = 5
    view = QtDims(Dims(ndim))
    qtbot.addWidget(view)

    for i, widg in enumerate(view.slider_widgets):
        widg.slider.sliderPressed.emit()
        assert view.last_used == i


def test_play_button(qtbot):
    """test that the play button and its popup dialog work"""
    ndim = 3
    view = QtDims(Dims(ndim))
    qtbot.addWidget(view)
    button = view.slider_widgets[0].play_button
    qtbot.mouseClick(button, Qt.LeftButton)
    qtbot.waitSignal(view._animation_thread.started, timeout=5000)
    qtbot.wait(200)
    assert view.is_playing
    with qtbot.waitSignal(view._animation_thread.finished, timeout=7000):
        qtbot.mouseClick(button, Qt.LeftButton)
    qtbot.wait(200)
    assert not view.is_playing

    assert not button.popup.isVisible()
    qtbot.mouseClick(button, Qt.RightButton)
    assert button.popup.isVisible()
