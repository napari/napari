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


def test_depiction_combobox_changes(qtbot):
    """Changing the model attribute should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    layer._slice_dims(ndisplay=3)
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo_box = qtctrl.depictionComboBox
    opts = {combo_box.itemText(i) for i in range(combo_box.count())}
    depiction_options = {
        'volume',
        'plane',
    }
    assert opts == depiction_options
    layer.depiction = 'plane'
    assert combo_box.findText('plane') == combo_box.currentIndex()
    layer.depiction = 'volume'
    assert combo_box.findText('volume') == combo_box.currentIndex()


def test_plane_controls_show_hide_on_depiction_change(qtbot):
    """Changing depiction mode should show/hide plane controls in 3D."""
    layer = Image(np.random.rand(10, 15, 20))
    layer._slice_dims(ndisplay=3)
    qtctrl = QtImageControls(layer)
    plane_controls = (
        qtctrl.planeNormalButtons,
        qtctrl.planeNormalLabel,
        qtctrl.planeThicknessSlider,
        qtctrl.planeThicknessLabel,
    )

    layer.depiction = 'volume'
    for widget in plane_controls:
        assert widget.isHidden()

    layer.depiction = 'plane'
    for widget in plane_controls:
        assert not widget.isHidden()  # isVisible() != not isHidden()


def test_plane_controls_show_hide_on_ndisplay_change(qtbot):
    """Changing ndisplay should show/hide plane controls if depicting a plane."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    plane_controls = (
        qtctrl.planeNormalButtons,
        qtctrl.planeNormalLabel,
        qtctrl.planeThicknessSlider,
        qtctrl.planeThicknessLabel,
    )

    layer._slice_dims(ndisplay=3)
    layer.depiction = 'plane'
    for widget in plane_controls:
        assert not widget.isHidden()  # isVisible() != not isHidden()

    layer._slice_dims(ndisplay=2)
    for widget in plane_controls:
        assert widget.isHidden()
