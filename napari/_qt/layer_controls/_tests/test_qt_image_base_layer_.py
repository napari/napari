import os
from unittest.mock import patch

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari._qt.layer_controls.qt_image_controls_base import (
    QRangeSliderPopup,
    QtBaseImageControls,
    create_range_popup,
)
from napari.layers import Image, Surface

_IMAGE = np.arange(100).astype(np.uint16).reshape((10, 10))
_SURF = (
    np.random.random((10, 2)),
    np.random.randint(10, size=(6, 3)),
    np.arange(100).astype(float),
)


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_base_controls_creation(qtbot, layer):
    """Check basic creation of QtBaseImageControls works"""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    original_clims = tuple(layer.contrast_limits)
    slider_clims = qtctrl.contrastLimitsSlider.values()
    assert slider_clims[0] == 0
    assert slider_clims[1] == 99
    assert tuple(slider_clims) == original_clims


@patch.object(QRangeSliderPopup, 'show')
@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_clim_right_click_shows_popup(mock_show, qtbot, layer):
    """Right clicking on the contrast limits slider should show a popup."""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtbot.mousePress(qtctrl.contrastLimitsSlider, Qt.RightButton)
    assert hasattr(qtctrl, 'clim_pop')
    # this mock doesn't seem to be working on cirrus windows
    # but it works on local windows tests...
    if not (os.name == 'nt' and os.getenv("CI")):
        mock_show.assert_called_once()


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_changing_model_updates_view(qtbot, layer):
    """Changing the model attribute should update the view"""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    new_clims = (20, 40)
    layer.contrast_limits = new_clims
    assert tuple(qtctrl.contrastLimitsSlider.values()) == new_clims


@patch.object(QRangeSliderPopup, 'show')
@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_range_popup_clim_buttons(mock_show, qtbot, layer):
    """The buttons in the clim_popup should adjust the contrast limits value"""
    qtctrl = QtBaseImageControls(layer)
    qtbot.addWidget(qtctrl)
    original_clims = tuple(layer.contrast_limits)
    layer.contrast_limits = (20, 40)
    qtbot.mousePress(qtctrl.contrastLimitsSlider, Qt.RightButton)

    # pressing the reset button returns the clims to the default values
    reset_button = qtctrl.clim_pop.findChild(QPushButton, "reset_clims_button")
    reset_button.click()
    qtbot.wait(20)
    assert tuple(qtctrl.contrastLimitsSlider.values()) == original_clims

    rangebtn = qtctrl.clim_pop.findChild(QPushButton, "full_clim_range_button")
    # the data we created above was uint16 for Image, and float for Surface
    # Surface will not have a "full range button"
    if np.issubdtype(layer.dtype, np.integer):
        rangebtn.click()
        qtbot.wait(20)
        assert tuple(layer.contrast_limits_range) == (0, 2 ** 16 - 1)
        assert tuple(qtctrl.contrastLimitsSlider.range()) == (0, 2 ** 16 - 1)
    else:
        assert rangebtn is None


@pytest.mark.parametrize('mag', [-12, -9, -3, 0, 2, 4, 6])
def test_clim_slider_step_size_and_precision(qtbot, mag):
    """Make sure the slider has a reasonable step size and precision.

    ...across a broad range of orders of magnitude.
    """
    layer = Image(np.random.rand(20, 20) / 10 ** mag)
    popup = create_range_popup(layer, 'contrast_limits')
    qtbot.addWidget(popup)
    # the range slider popup labels should have a number of decimal points that
    # is inversely proportional to the order of magnitude of the range of data,
    # but should never be greater than 5 or less than 0
    assert popup.precision == max(min(mag + 3, 5), 0)

    # the slider step size should also be inversely proportional to the data
    # range, with 1000 steps across the data range
    assert np.ceil(popup.slider._step * 10 ** (mag + 4)) == 10
