import numpy as np
import pytest

from napari._qt.qt_range_slider_popup import QRangeSliderPopup
from napari._qt.utils import find_ancestor_mainwindow

import napari

initial = np.array([100, 400])
range_ = np.array([0, 500])


@pytest.fixture
def popup():
    popup = QRangeSliderPopup(
        horizontal=True,
        precision=2,
        initial_values=initial,
        data_range=range_,
        step_size=1,
    )
    popup.show()
    return popup


def test_range_slider_popup_labels(qtbot, popup):
    """make sure labels are correct"""
    assert float(popup.curmin_label.text()) == initial[0]
    assert float(popup.curmax_label.text()) == initial[1]
    assert np.all(popup.slider.range() == range_)


def test_range_slider_changes_labels(qtbot, popup):
    """make sure setting the slider updates the labels"""
    popup.slider.setValues((10, 20))
    assert float(popup.curmin_label.text()) == 10
    assert float(popup.curmax_label.text()) == 20


def test_labels_change_range_slider(qtbot, popup):
    """make sure setting the labels updates the slider"""
    popup.slider.setValues((10, 20))

    popup.curmin_label.setText('100')
    popup.curmax_label.setText('300')
    popup.curmin_label.editingFinished.emit()
    popup.curmax_label.editingFinished.emit()
    assert np.all(popup.slider.values() == (100, 300))


def test_find_main_window(qtbot):
    """Make sure find_ancestor_mainwindow works in a multi viewer environment.
    """
    v1 = napari.Viewer()
    layer1 = v1.add_image(np.random.rand(10, 10))
    v2 = napari.Viewer()
    layer2 = v2.add_image(np.random.rand(10, 10))

    ctrl1 = v1.window.qt_viewer.controls.widgets[layer1]
    ctrl2 = v2.window.qt_viewer.controls.widgets[layer2]
    assert find_ancestor_mainwindow(ctrl1) == v1.window._qt_window
    assert find_ancestor_mainwindow(ctrl2) == v2.window._qt_window
