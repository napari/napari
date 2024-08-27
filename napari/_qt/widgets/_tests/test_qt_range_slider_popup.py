import pytest

from napari._qt.widgets.qt_range_slider_popup import QRangeSliderPopup

initial = (100, 400)
range_ = (0, 500)


@pytest.fixture
def popup(qtbot):
    popup = QRangeSliderPopup()
    popup.slider.setRange(*range_)
    popup.slider.setValue(initial)
    qtbot.addWidget(popup)
    return popup


def test_range_slider_popup_labels(popup):
    """make sure labels are correct"""
    assert popup.slider._handle_labels[0].value() == initial[0]
    assert popup.slider._handle_labels[1].value() == initial[1]
    assert (popup.slider.minimum(), popup.slider.maximum()) == range_


def test_range_slider_changes_labels(popup):
    """make sure setting the slider updates the labels"""
    popup.slider.setValue((10, 20))
    assert popup.slider._handle_labels[0].value() == 10
    assert popup.slider._handle_labels[1].value() == 20
