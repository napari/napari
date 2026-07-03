import numpy as np
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari.components.dims import Dims
from napari.layers import Image


def test_interpolation_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl._interpolation_control.interpolation_combobox
    opts = {combo.itemText(i) for i in range(combo.count())}
    assert opts == {'cubic', 'linear', 'kaiser', 'nearest', 'spline36'}
    # programmatically adding approved interpolation works
    layer.interpolation2d = 'lanczos'
    assert combo.findText('lanczos') == 5


def test_rendering_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl._render_control.render_combobox
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
    qtctrl = QtImageControls(layer)
    qtctrl.ndisplay = 3
    qtbot.addWidget(qtctrl)
    combo_box = qtctrl._depiction_control.depiction_combobox
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
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    qtctrl.ndisplay = 3

    layer.depiction = 'volume'
    assert qtctrl._depiction_control.plane_thickness_slider.isHidden()
    assert qtctrl._depiction_control.plane_thickness_label.isHidden()
    assert qtctrl._depiction_control.plane_normal_buttons.isHidden()
    assert qtctrl._depiction_control.plane_normal_label.isHidden()

    layer.depiction = 'plane'
    assert not qtctrl._depiction_control.plane_thickness_slider.isHidden()
    assert not qtctrl._depiction_control.plane_thickness_label.isHidden()
    assert not qtctrl._depiction_control.plane_normal_buttons.isHidden()
    assert not qtctrl._depiction_control.plane_normal_label.isHidden()


def test_plane_controls_show_hide_on_ndisplay_change(qtbot):
    """Changing ndisplay should show/hide plane controls if depicting a plane."""
    layer = Image(np.random.rand(10, 15, 20))
    layer.depiction = 'plane'
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    assert qtctrl.ndisplay == 2
    assert qtctrl._depiction_control.plane_thickness_slider.isHidden()
    assert qtctrl._depiction_control.plane_thickness_label.isHidden()
    assert qtctrl._depiction_control.plane_normal_buttons.isHidden()
    assert qtctrl._depiction_control.plane_normal_label.isHidden()

    qtctrl.ndisplay = 3
    assert not qtctrl._depiction_control.plane_thickness_slider.isHidden()
    assert not qtctrl._depiction_control.plane_thickness_label.isHidden()
    assert not qtctrl._depiction_control.plane_normal_buttons.isHidden()
    assert not qtctrl._depiction_control.plane_normal_label.isHidden()


def test_plane_slider_value_change(qtbot):
    """Changing the model should update the view."""
    layer = Image(np.random.rand(10, 15, 20))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    layer.plane.thickness *= 2
    assert (
        qtctrl._depiction_control.plane_thickness_slider.value()
        == layer.plane.thickness
    )


def test_auto_contrast_buttons(qtbot):
    layer = Image(np.arange(8**3).reshape(8, 8, 8), contrast_limits=(0, 1))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)
    assert layer.contrast_limits == [0, 1]
    qtctrl._contrast_limits_control.auto_scale_bar._once_btn.click()
    assert layer.contrast_limits == [0, 63]

    # change slice
    dims = Dims(
        ndim=3, range=((0, 4, 1), (0, 8, 1), (0, 8, 1)), point=(1, 8, 8)
    )
    layer._slice_dims(dims)
    # hasn't changed yet
    assert layer.contrast_limits == [0, 63]

    # with auto_btn, it should always change
    qtctrl._contrast_limits_control.auto_scale_bar._auto_btn.click()
    assert layer.contrast_limits == [64, 127]
    dims.point = (2, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [128, 191]
    dims.point = (3, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [192, 255]

    # once button turns off continuous
    qtctrl._contrast_limits_control.auto_scale_bar._once_btn.click()
    dims.point = (4, 8, 8)
    layer._slice_dims(dims)
    assert layer.contrast_limits == [192, 255]


def test_histogram_button_toggles_inline_histogram(qtbot):
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    button = qtctrl._contrast_limits_control.histogram_button
    assert button is not None
    assert qtctrl._histogram_control is not None
    assert qtctrl._histogram_control.content_widget.isHidden()

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert not qtctrl._histogram_control.content_widget.isHidden()
    assert (
        qtctrl.layout().labelForField(qtctrl._histogram_control.content_widget)
        is None
    )
    assert layer.histogram.enabled
    assert button.isChecked()

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert qtctrl._histogram_control.content_widget.isHidden()
    assert not layer.histogram.enabled
    assert not button.isChecked()


def test_histogram_button_right_click_opens_popup(qtbot):
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    button = qtctrl._contrast_limits_control.histogram_button
    assert button is not None

    qtbot.mouseClick(button, Qt.MouseButton.RightButton)

    popup = qtctrl._contrast_limits_control.clim_popup
    assert popup is not None
    assert popup.histogram_content is not None
    assert popup.histogram_content.histogram_widget is not None
    assert popup.histogram_content.settings_widget is not None
    assert not button.isChecked()

    popup.close()


def test_histogram_control_lazy_creation(qtbot):
    """Histogram control should lazily create content on first ensure_content() call."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    # Before ensure_content: content_widget exists but histogram_content is None
    assert qtctrl._histogram_control is not None
    assert qtctrl._histogram_control.content_widget is not None
    assert qtctrl._histogram_control.histogram_content is None
    assert qtctrl._histogram_control.histogram_widget is None
    assert qtctrl._histogram_control.settings_widget is None

    # After ensure_content: all sub-widgets exist
    qtctrl._histogram_control.ensure_content()
    assert qtctrl._histogram_control.histogram_content is not None
    assert qtctrl._histogram_control.histogram_widget is not None
    assert qtctrl._histogram_control.settings_widget is not None

    # Second call is idempotent
    qtctrl._histogram_control.ensure_content()
    assert qtctrl._histogram_control.histogram_content is not None


def test_histogram_widget_responds_to_viewer_theme_toggle(
    qtbot, make_napari_viewer
):
    """viewer.theme change (Ctrl+Shift+T) should repaint histogram canvas.

    The ``toggle_theme`` keybinding sets ``viewer.theme`` directly without
    touching ``settings.appearance.theme``. The container bridges this to
    histogram widgets — verify the canvas repaints with the right colors.
    """
    from napari.utils.theme import get_theme

    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    )
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    controls._histogram_control.ensure_content()
    widget = controls._histogram_control.histogram_widget
    assert widget is not None

    layer.histogram.enabled = True
    layer.histogram.compute()

    # Pick a theme different from the current one, same as Ctrl+Shift+T.
    new_theme = 'light' if viewer.theme != 'light' else 'dark'
    viewer.theme = new_theme
    expected = get_theme(new_theme)

    qtbot.waitUntil(
        lambda: np.allclose(
            widget.canvas.bgcolor.rgba[:3],
            np.array(expected.canvas.as_rgb_tuple()) / 255,
        )
    )


def test_two_image_layers_independent_histograms(qtbot, make_napari_viewer):
    """Two Image layers should each have their own independent histogram."""
    viewer = make_napari_viewer()
    layer_a = viewer.add_image(np.random.rand(10, 10), name='a')
    layer_b = viewer.add_image(np.random.rand(10, 10), name='b')

    # Each layer has its own histogram model
    assert layer_a.histogram is not layer_b.histogram

    # Each has its own controls with histogram
    controls_a = viewer.window._qt_viewer.controls.widgets[layer_a]
    controls_b = viewer.window._qt_viewer.controls.widgets[layer_b]
    assert controls_a._histogram_control is not None
    assert controls_b._histogram_control is not None
    assert controls_a._histogram_control is not controls_b._histogram_control

    # Toggling one doesn't affect the other
    button_a = controls_a._contrast_limits_control.histogram_button
    qtbot.mouseClick(button_a, Qt.MouseButton.LeftButton)
    assert layer_a.histogram.enabled
    assert not layer_b.histogram.enabled
