import numpy as np

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari.layers import Image


def test_interpolation_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.interpComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    assert opts == {'bicubic', 'bilinear', 'kaiser', 'nearest', 'spline36'}
    # programmatically adding approved interpolation works
    layer.interpolation = 'lanczos'
    assert combo.findText('lanczos') == 5


def test_rendering_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.renderComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {
        'translucent',
        'additive',
        'iso',
        'mip',
        'minip',
        'attenuated_mip',
        'average',
    }
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    layer.rendering = 'iso'
    assert combo.findText('iso') == combo.currentIndex()


def test_auto_contrast_buttons(qtbot):
    layer = Image(np.arange(8**3).reshape(8, 8, 8), contrast_limits=(0, 1))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    assert layer.contrast_limits == [0, 1]
    qtctrl.autoScaleBar._once_btn.click()
    assert layer.contrast_limits == [0, 63]

    # change slice
    layer._slice_dims((1, 8, 8))
    # hasn't changed yet
    assert layer.contrast_limits == [0, 63]

    # with auto_btn, it should always change
    qtctrl.autoScaleBar._auto_btn.click()
    assert layer.contrast_limits == [64, 127]
    layer._slice_dims((2, 8, 8))
    assert layer.contrast_limits == [128, 191]
    layer._slice_dims((3, 8, 8))
    assert layer.contrast_limits == [192, 255]

    # once button turns off continuous
    qtctrl.autoScaleBar._once_btn.click()
    layer._slice_dims((4, 8, 8))
    assert layer.contrast_limits == [192, 255]
