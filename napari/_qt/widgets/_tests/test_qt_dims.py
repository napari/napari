import os
from sys import platform
from unittest.mock import patch

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims


def test_creating_view(qtbot):
    """
    Test creating dims view.
    """
    ndim = 4
    dims = Dims(ndim=ndim)
    view = QtDims(dims)

    qtbot.addWidget(view)

    # Check that the dims model has been appended to the dims view
    assert view.dims == dims

    # Check the number of displayed sliders is two less than the number of
    # dimensions
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )


def test_changing_ndim(qtbot):
    """
    Test changing the number of dimensions
    """
    ndim = 4
    view = QtDims(Dims(ndim=ndim))

    qtbot.addWidget(view)

    # Check that adding dimensions adds sliders
    view.dims.ndim = 5
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )

    # Check that removing dimensions removes sliders
    view.dims.ndim = 2
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )


def test_changing_focus(qtbot):
    """Test changing focus updates the dims.last_used prop."""
    # Initialize to 0th axis
    ndim = 2
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)
    assert view.dims.last_used == 0
    view.dims._focus_down()
    view.dims._focus_up()
    assert view.dims.last_used == 0

    view.dims.ndim = 5
    view.dims.last_used = 2
    assert view.dims.last_used == 2
    view.dims._focus_down()
    assert view.dims.last_used == 1
    view.dims._focus_up()
    assert view.dims.last_used == 2
    view.dims._focus_up()
    assert view.dims.last_used == 0
    view.dims._focus_down()
    assert view.dims.last_used == 2


def test_changing_display(qtbot):
    """
    Test changing the displayed property of an axis
    """
    ndim = 4
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)

    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 2
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )

    # Check changing displayed removes a slider
    view.dims.ndisplay = 3
    assert view.nsliders == view.dims.ndim
    assert np.sum(view._displayed_sliders) == view.dims.ndim - 3
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )


def test_slider_values(qtbot):
    """
    Test the values of a slider stays matched to the values of the dims point.
    """
    ndim = 4
    view = QtDims(Dims(ndim=ndim))
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
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)

    # Check the maximum allowed value of the slider is one less
    # than the allowed nsteps of the dims at initialization
    first_slider = view.slider_widgets[0].slider
    assert first_slider.minimum() == 0
    assert first_slider.maximum() == view.dims.nsteps[0] - 1
    assert first_slider.singleStep() == 1

    # Check the maximum allowed value of the slider stays one less
    # than the allowed nsteps of the dims after updates
    view.dims.set_range(0, (1, 5, 2))
    assert first_slider.minimum() == 0
    assert first_slider.maximum() == view.dims.nsteps[0] - 1
    assert first_slider.singleStep() == 1


def test_singleton_dims(qtbot):
    """
    Test singleton dims causes no slider.
    """
    ndim = 4
    dims = Dims(ndim=ndim)
    dims.set_range(0, (0, 1, 1))
    view = QtDims(dims)
    qtbot.addWidget(view)

    # Check that the dims model has been appended to the dims view
    assert view.dims == dims

    # Check the number of displayed sliders is only one
    assert view.nsliders == 4
    assert np.sum(view._displayed_sliders) == 1
    assert np.all(
        [
            s.isVisibleTo(view) == d
            for s, d in zip(view.slider_widgets, view._displayed_sliders)
        ]
    )

    # Change ndisplay to three
    view.dims.ndisplay = 3

    # Check no sliders now shown
    assert np.sum(view._displayed_sliders) == 0

    # Change ndisplay back to two
    view.dims.ndisplay = 2

    # Check only slider now shown
    assert np.sum(view._displayed_sliders) == 1


def test_order_when_changing_ndim(qtbot):
    """
    Test order of the sliders when changing the number of dimensions.
    """
    ndim = 4
    view = QtDims(Dims(ndim=ndim))
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
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)
    view.dims.axis_labels = list('TZYX')
    assert [w.axis_label.text() for w in view.slider_widgets] == list('TZYX')

    observed_axis_labels_event = False

    def on_axis_labels_changed():
        nonlocal observed_axis_labels_event
        observed_axis_labels_event = True

    view.dims.events.axis_labels.connect(on_axis_labels_changed)
    first_label = view.slider_widgets[0].axis_label
    assert first_label.text() == view.dims.axis_labels[0]
    first_label.setText('napari')
    assert first_label.text() == view.dims.axis_labels[0]
    assert observed_axis_labels_event


def test_slider_press_updates_last_used(qtbot):
    """pressing on the slider should update the dims.last_used property"""
    ndim = 5
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)

    for i, widg in enumerate(view.slider_widgets):
        widg.slider.sliderPressed.emit()
        assert view.dims.last_used == i


@pytest.mark.skipif(
    os.environ.get('CI') and platform == 'win32',
    reason='not working in windows VM',
)
def test_play_button(qtbot):
    """test that the play button and its popup dialog work"""
    ndim = 3
    view = QtDims(Dims(ndim=ndim))
    qtbot.addWidget(view)
    slider = view.slider_widgets[0]
    button = slider.play_button

    # Need looping playback so that it does not stop before we can assert that.
    assert slider.loop_mode == 'loop'
    assert not view.is_playing

    qtbot.mouseClick(button, Qt.LeftButton)
    qtbot.waitUntil(lambda: view.is_playing)

    qtbot.mouseClick(button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not view.is_playing)
    qtbot.waitUntil(lambda: view._animation_worker is None)

    with patch.object(button.popup, 'show_above_mouse') as mock_popup:
        qtbot.mouseClick(button, Qt.RightButton)
        mock_popup.assert_called_once()


def test_slice_labels(qtbot):
    ndim = 4
    dims = Dims(ndim=ndim)
    dims.set_range(0, (0, 20, 1))
    view = QtDims(dims)
    qtbot.addWidget(view)

    # make sure the totslice_label is showing the correct number
    assert int(view.slider_widgets[0].totslice_label.text()) == 19

    # make sure setting the dims.point updates the slice label
    label_edit = view.slider_widgets[0].curslice_label
    dims.set_point(0, 15)
    assert int(label_edit.text()) == 15

    # make sure setting the current slice label updates the model
    label_edit.setText(str(8))
    label_edit.editingFinished.emit()
    assert dims.point[0] == 8


def test_not_playing_after_ndim_changes(qtbot):
    """See https://github.com/napari/napari/issues/3998"""
    dims = Dims(ndim=3, ndisplay=2, range=((0, 10, 1), (0, 20, 1), (0, 30, 1)))
    view = QtDims(dims)
    qtbot.addWidget(view)
    # Loop to prevent finishing before the assertions in this test.
    view.play(loop_mode='loop')
    qtbot.waitUntil(lambda: view.is_playing)

    dims.ndim = 2

    qtbot.waitUntil(lambda: not view.is_playing)
    qtbot.waitUntil(lambda: view._animation_worker is None)


def test_not_playing_after_ndisplay_changes(qtbot):
    """See https://github.com/napari/napari/issues/3998"""
    dims = Dims(ndim=3, ndisplay=2, range=((0, 10, 1), (0, 20, 1), (0, 30, 1)))
    view = QtDims(dims)
    qtbot.addWidget(view)
    # Loop to prevent finishing before the assertions in this test.
    view.play(loop_mode='loop')
    qtbot.waitUntil(lambda: view.is_playing)

    dims.ndisplay = 3

    qtbot.waitUntil(lambda: not view.is_playing)
    qtbot.waitUntil(lambda: view._animation_worker is None)


def test_set_axis_labels_after_ndim_changes(qtbot):
    """See https://github.com/napari/napari/issues/3753"""
    dims = Dims(ndim=3, ndisplay=2)
    view = QtDims(dims)
    qtbot.addWidget(view)

    dims.ndim = 2
    dims.axis_labels = ['y', 'x']

    assert len(view.slider_widgets) == 2
    assert view.slider_widgets[0].axis_label.text() == 'y'
    assert view.slider_widgets[1].axis_label.text() == 'x'
