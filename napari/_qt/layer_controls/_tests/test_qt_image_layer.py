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
    layer.interpolation2d = 'lanczos'
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
    qtbot.addWidget(qtctrl)

    layer.depiction = 'volume'
    assert qtctrl.planeThicknessSlider.isHidden()
    assert qtctrl.planeThicknessLabel.isHidden()
    assert qtctrl.planeNormalButtons.isHidden()
    assert qtctrl.planeNormalLabel.isHidden()

    layer.depiction = 'plane'
    assert not qtctrl.planeThicknessSlider.isHidden()
    assert not qtctrl.planeThicknessLabel.isHidden()
    assert not qtctrl.planeNormalButtons.isHidden()
    assert not qtctrl.planeNormalLabel.isHidden()


def test_plane_controls_show_hide_on_ndisplay_change(qtbot):
    """Changing ndisplay should show/hide plane controls if depicting a plane."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    layer._slice_dims(ndisplay=3)
    layer.depiction = 'plane'
    assert not qtctrl.planeThicknessSlider.isHidden()
    assert not qtctrl.planeThicknessLabel.isHidden()
    assert not qtctrl.planeNormalButtons.isHidden()
    assert not qtctrl.planeNormalLabel.isHidden()

    layer._slice_dims(ndisplay=2)
    assert qtctrl.planeThicknessSlider.isHidden()
    assert qtctrl.planeThicknessLabel.isHidden()
    assert qtctrl.planeNormalButtons.isHidden()
    assert qtctrl.planeNormalLabel.isHidden()


def test_plane_slider_value_change(qtbot):
    """Changing the model should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    layer.plane.thickness *= 2
    assert qtctrl.planeThicknessSlider.value() == layer.plane.thickness


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
