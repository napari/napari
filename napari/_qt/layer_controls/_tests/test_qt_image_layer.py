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


def test_depiction_combobox(qtbot):
    """Changing the model attribute should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo_box = qtctrl.depictionComboBox
    opts = {combo_box.itemText(i) for i in range(combo_box.count())}
    depiction_options = {
        'volume',
        'plane',
    }
    assert opts == depiction_options
    layer.depiction == 'plane'
    assert combo_box.findText('plane') == combo_box.currentIndex()
    layer.depiction == 'volume'
    assert combo_box.findText('volume') == combo_box.currentIndex()
