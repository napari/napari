"""
Suspecting Segfaulting test.

The test on this file have been put in their own file to try to narrow down a
Sega fluting test.

As when test segfault, pytest output get corrupted, it is hard to know
which of the test the previous file is segfaulting.

Moving the test here, at least allow us to know in which file the segfaulting
happens as _at least_ the file name will be printed.
"""

from napari._qt.widgets.qt_dims import QtDims
from napari.components import Dims


def test_slice_labels(qtbot):
    ndim = 4
    dims = Dims(ndim=ndim)
    dims.set_range(0, (0, 20, 1))
    view = QtDims(dims)
    qtbot.addWidget(view)

    # make sure the totslice_label is showing the correct number
    assert int(view.slider_widgets[0].totslice_label.text()) == 20

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
