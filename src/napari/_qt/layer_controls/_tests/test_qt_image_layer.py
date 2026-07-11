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

    # Enable histogram first so the popup lazy-creates histogram content
    layer.histogram.enabled = True

    qtbot.mouseClick(button, Qt.MouseButton.RightButton)

    popup = qtctrl._contrast_limits_control.clim_popup
    assert popup is not None
    # Histogram content is lazy-created in showEvent; call _ensure first
    popup._ensure_histogram_content()
    assert popup.histogram_content is not None
    assert popup.histogram_content.histogram_widget is not None
    assert popup.histogram_content.settings_widget is not None
    assert button.isChecked()  # enabled=True syncs the button

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


def test_histogram_popup_and_inline_coexistence(qtbot, make_napari_viewer):
    """Opening the histogram popup while inline histogram is showing should work.

    Both the inline (layer controls) and popup histogram should be functional
    without conflicting. Closing the popup should not disable the inline histogram.
    """
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.rand(10, 10))
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    control = controls._histogram_control
    assert control is not None

    button = controls._contrast_limits_control.histogram_button
    assert button is not None

    # 1. Enable inline histogram via left-click
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert not control.content_widget.isHidden()
    assert layer.histogram.enabled

    # 2. Open popup via right-click (while inline is showing)
    qtbot.mouseClick(button, Qt.MouseButton.RightButton)
    popup = controls._contrast_limits_control.clim_popup
    assert popup is not None
    # Histogram content is lazy-created in showEvent since enabled=True
    assert popup.histogram_content is not None

    # The popup's histogram should have its own content widget instance
    assert popup.histogram_content is not control.histogram_content

    # 3. Close popup — inline histogram should still be enabled
    popup.close()
    assert layer.histogram.enabled
    assert not control.content_widget.isHidden()

    # 4. Toggle inline off
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert control.content_widget.isHidden()
    assert not layer.histogram.enabled


def test_api_enable_shows_inline_widget(qtbot):
    """``layer.histogram.enabled = True`` via API should show the inline content_widget."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    control = qtctrl._histogram_control
    assert control is not None
    assert control.content_widget.isHidden()

    # API enable — should show widget and trigger computation
    layer.histogram.enabled = True

    assert not control.content_widget.isHidden()
    assert control.histogram_content is not None
    # Bin edges should have been computed
    assert len(layer.histogram._bin_edges) == 257

    # API disable — should hide widget
    layer.histogram.enabled = False
    assert control.content_widget.isHidden()


def test_api_enable_syncs_button_checked_state(qtbot):
    """``layer.histogram.enabled`` changes via API should sync the button's checked state."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    button = qtctrl._contrast_limits_control.histogram_button
    assert button is not None
    assert not button.isChecked()

    # API enable — button should become checked
    layer.histogram.enabled = True
    assert button.isChecked()

    # API disable — button should become unchecked
    layer.histogram.enabled = False
    assert not button.isChecked()


def test_popup_does_not_include_histogram_when_disabled(qtbot):
    """Right-click popup's histogram content should be hidden when ``enabled`` is False.

    The histogram content widget is always present for Image layers (to avoid
    layout blink on toggle), but starts hidden.  Inline state must remain
    unaffected by opening the popup.
    """
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    control = qtctrl._histogram_control
    button = qtctrl._contrast_limits_control.histogram_button
    assert control.content_widget.isHidden()
    assert not button.isChecked()

    # Right-click to open popup — histogram is disabled, so popup
    # histogram content should be lazy-created but hidden.
    qtbot.mouseClick(button, Qt.MouseButton.RightButton)

    popup = qtctrl._contrast_limits_control.clim_popup
    assert popup is not None
    # Content not yet created (disabled, no showEvent trigger)
    assert popup.histogram_content is None

    # Inline widget should not have been affected
    assert control.content_widget.isHidden()
    assert not button.isChecked()

    # Close popup — inline state should still be unchanged
    popup.close()
    assert control.content_widget.isHidden()
    assert not button.isChecked()


def test_popup_does_not_disable_inline_histogram(qtbot):
    """Popup should not disable an already-enabled inline histogram on close."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    # Enable inline histogram first
    layer.histogram.enabled = True
    assert not qtctrl._histogram_control.content_widget.isHidden()
    assert qtctrl._contrast_limits_control.histogram_button.isChecked()

    # Open popup via right-click
    qtbot.mouseClick(
        qtctrl._contrast_limits_control.histogram_button,
        Qt.MouseButton.RightButton,
    )
    popup = qtctrl._contrast_limits_control.clim_popup
    assert popup is not None
    # Histogram content is lazy-created in showEvent since enabled=True
    assert popup.histogram_content is not None

    # Close popup — inline histogram should still be enabled
    popup.close()
    assert layer.histogram.enabled
    assert not qtctrl._histogram_control.content_widget.isHidden()
    assert qtctrl._contrast_limits_control.histogram_button.isChecked()


def test_popup_histogram_checkbox_toggle(qtbot):
    """Popup histogram checkbox should lazy-create and show/hide histogram content."""
    layer = Image(np.random.rand(8, 8))
    qtctrl = QtImageControls(layer)
    qtbot.addWidget(qtctrl)

    # Open popup
    qtbot.mouseClick(
        qtctrl._contrast_limits_control.histogram_button,
        Qt.MouseButton.RightButton,
    )
    popup = qtctrl._contrast_limits_control.clim_popup
    assert popup is not None
    assert popup._histogram_enabled_checkbox is not None

    # Histogram content not yet created (disabled by default)
    assert popup.histogram_content is None
    assert not popup._histogram_enabled_checkbox.isChecked()

    # Check the checkbox — histogram should be lazy-created and show
    popup._histogram_enabled_checkbox.setChecked(True)
    qtbot.waitUntil(lambda: popup.histogram_content is not None)
    qtbot.waitUntil(lambda: not popup.histogram_content.isHidden())
    assert layer.histogram.enabled

    # Uncheck — histogram should hide
    popup._histogram_enabled_checkbox.setChecked(False)
    qtbot.waitUntil(lambda: popup.histogram_content.isHidden())
    assert not layer.histogram.enabled

    popup.close()
