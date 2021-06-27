import numpy as np
import pytest

from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup

initial = np.array([100, 400])
range_ = np.array([0, 500])


@pytest.fixture
def popup(qtbot):
    popup = QRangeSliderPopup(
        horizontal=True,
        precision=2,
        initial_values=initial,
        data_range=range_,
        step_size=1,
    )
    qtbot.addWidget(popup)
    return popup


def test_range_slider_popup_labels(popup):
    """make sure labels are correct"""
    assert float(popup.curmin_label.text()) == initial[0]
    assert float(popup.curmax_label.text()) == initial[1]
    assert np.all(popup.slider.range() == range_)


def test_range_slider_changes_labels(popup):
    """make sure setting the slider updates the labels"""
    popup.slider.setValues((10, 20))
    assert float(popup.curmin_label.text()) == 10
    assert float(popup.curmax_label.text()) == 20


def test_labels_change_range_slider(popup):
    """make sure setting the labels updates the slider"""
    popup.slider.setValues((10, 20))

    popup.curmin_label.setText('100')
    popup.curmax_label.setText('300')
    popup.curmin_label.editingFinished.emit()
    popup.curmax_label.editingFinished.emit()
    assert np.all(popup.slider.values() == (100, 300))
