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
