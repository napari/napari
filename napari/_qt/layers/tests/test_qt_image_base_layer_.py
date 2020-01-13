import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from napari.layers import Image, Surface
from napari._qt.layers.qt_image_base_layer import QtBaseImageControls


_IMAGE = np.arange(100).astype(np.uint16).reshape((10, 10))
_SURF = (
    np.random.random((10, 2)),
    np.random.randint(10, size=(6, 3)),
    np.arange(100).astype(np.float),
)


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_base_controls_creation(qtbot, layer):
    """Check basic creation of QtBaseImageControls works"""
    qtctrl = QtBaseImageControls(layer)
    original_clims = tuple(layer.contrast_limits)
    slider_clims = qtctrl.contrastLimitsSlider.values()
    assert slider_clims[0] == 0
    assert slider_clims[1] == 99
    assert tuple(slider_clims) == original_clims


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_clim_right_click_shows_popup(qtbot, layer):
    """Right clicking on the contrast limits slider should show a popup."""
    qtctrl = QtBaseImageControls(layer)
    qtbot.mousePress(qtctrl.contrastLimitsSlider, Qt.RightButton)
    assert hasattr(qtctrl, 'clim_pop')
    assert qtctrl.clim_pop.isVisible()


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_changing_model_updates_view(qtbot, layer):
    """Changing the model attribute should update the view"""
    qtctrl = QtBaseImageControls(layer)
    new_clims = (20, 40)
    layer.contrast_limits = new_clims
    assert tuple(qtctrl.contrastLimitsSlider.values()) == new_clims


@pytest.mark.parametrize('layer', [Image(_IMAGE), Surface(_SURF)])
def test_range_popup_clim_buttons(qtbot, layer):
    """The buttons in the clim_popup should adjust the contrast limits value"""
    qtctrl = QtBaseImageControls(layer)
    original_clims = tuple(layer.contrast_limits)
    layer.contrast_limits = (20, 40)
    qtbot.mousePress(qtctrl.contrastLimitsSlider, Qt.RightButton)

    # pressing the reset button returns the clims to the default values
    reset_button = qtctrl.clim_pop.findChild(QPushButton, "reset_clims_button")
    qtbot.mouseClick(reset_button, Qt.LeftButton)
    assert tuple(qtctrl.contrastLimitsSlider.values()) == original_clims

    rangebtn = qtctrl.clim_pop.findChild(QPushButton, "full_clim_range_button")
    # the data we created above was uint16 for Image, and float for Surface
    # Surface will not have a "full range button"
    if np.issubdtype(layer.dtype, np.integer):
        qtbot.mouseClick(rangebtn, Qt.LeftButton)
        assert tuple(layer.contrast_limits_range) == (0, 2 ** 16 - 1)
        assert tuple(qtctrl.contrastLimitsSlider.range()) == (0, 2 ** 16 - 1)
    else:
        assert rangebtn is None
