import numpy as np
import pytest

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.layers.labels._labels_constants import (
    IsoCategoricalGradientMode,
    LabelsRendering,
)
from napari.utils.colormaps import DirectLabelColormap, colormap_utils

np.random.seed(0)
_LABELS = np.random.randint(5, size=(10, 15), dtype=np.uint8)
_COLOR = DirectLabelColormap(
    color_dict={
        1: 'white',
        2: 'blue',
        3: 'green',
        4: 'red',
        5: 'yellow',
        None: 'black',
    }
)


@pytest.fixture
def make_labels_controls(qtbot, colormap=None):
    def _make_labels_controls(colormap=colormap):
        layer = Labels(_LABELS, colormap=colormap)
        qtctrl = QtLabelsControls(layer)
        qtbot.add_widget(qtctrl)
        return layer, qtctrl

    return _make_labels_controls


def test_changing_layer_color_mode_updates_combo_box(make_labels_controls):
    """Updating layer color mode changes the combo box selection"""
    layer, qtctrl = make_labels_controls(colormap=_COLOR)

    assert (
        qtctrl._colormode_combobox_control.color_mode_combobox.currentText()
        == 'direct'
    )

    layer.colormap = layer._random_colormap
    assert (
        qtctrl._colormode_combobox_control.color_mode_combobox.currentText()
        == 'auto'
    )


def test_changing_layer_show_selected_label_updates_check_box(
    make_labels_controls,
):
    """See https://github.com/napari/napari/issues/5371"""
    layer, qtctrl = make_labels_controls()
    assert not qtctrl._display_selected_label_checkbox_control.selected_color_checkbox.isChecked()
    assert not layer.show_selected_label

    layer.show_selected_label = True

    assert qtctrl._display_selected_label_checkbox_control.selected_color_checkbox.isChecked()


def test_rendering_combobox(make_labels_controls):
    """Changing the model attribute should update the view"""
    layer, qtctrl = make_labels_controls()
    combo = qtctrl._render_control.render_combobox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {'translucent', 'iso_categorical'}
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    new_mode = 'iso_categorical'
    layer.rendering = new_mode
    assert combo.findText(new_mode) == combo.currentIndex()


def test_changing_colormap_updates_colorbox(make_labels_controls):
    """Test that changing the colormap on a layer will update color swatch in the combo box"""
    layer, qtctrl = make_labels_controls(colormap=_COLOR)
    color_box = qtctrl._label_control.colorbox

    layer.selected_label = 1

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(
        color_box.color,
        np.round(np.asarray(layer._selected_color) * 255),
    )

    layer.colormap = colormap_utils.label_colormap(num_colors=5)

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(
        color_box.color,
        np.round(np.asarray(layer._selected_color) * 255),
    )


def test_selected_color_checkbox(make_labels_controls):
    """Tests that the 'selected color' checkbox sets the 'show_selected_label' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl._display_selected_label_checkbox_control.selected_color_checkbox.setChecked(
        True
    )
    assert layer.show_selected_label
    qtctrl._display_selected_label_checkbox_control.selected_color_checkbox.setChecked(
        False
    )
    assert not layer.show_selected_label
    qtctrl._display_selected_label_checkbox_control.selected_color_checkbox.setChecked(
        True
    )
    assert layer.show_selected_label


def test_contiguous_labels_checkbox(make_labels_controls):
    """Tests that the 'contiguous' checkbox sets the 'contiguous' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl._contiguous_checkbox_control.contiguous_checkbox.setChecked(True)
    assert layer.contiguous
    qtctrl._contiguous_checkbox_control.contiguous_checkbox.setChecked(False)
    assert not layer.contiguous
    qtctrl._contiguous_checkbox_control.contiguous_checkbox.setChecked(True)
    assert layer.contiguous


def test_preserve_labels_checkbox(make_labels_controls):
    """Tests that the 'preserve labels' checkbox sets the 'preserve_labels' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl._preserve_labels_checkbox_control.preserve_labels_checkbox.setChecked(
        True
    )
    assert layer.preserve_labels
    qtctrl._preserve_labels_checkbox_control.preserve_labels_checkbox.setChecked(
        False
    )
    assert not layer.preserve_labels
    qtctrl._preserve_labels_checkbox_control.preserve_labels_checkbox.setChecked(
        True
    )
    assert layer.preserve_labels


def test_change_label_selector_range(make_labels_controls):
    """Changing the label layer dtype should update label selector range."""
    layer, qtctrl = make_labels_controls()
    assert layer.data.dtype == np.uint8
    assert qtctrl._label_control.selection_spinbox.minimum() == 0
    assert qtctrl._label_control.selection_spinbox.maximum() == 255

    layer.data = layer.data.astype(np.int8)

    assert qtctrl._label_control.selection_spinbox.minimum() == -128
    assert qtctrl._label_control.selection_spinbox.maximum() == 127


def test_initial_label_selector_value(make_labels_controls):
    """Initializing the label selector spinbox to an initial value."""
    layer, qtctrl = make_labels_controls()
    assert (
        qtctrl._label_control.selection_spinbox.value() == layer.selected_label
    )


def test_change_iso_gradient_mode(make_labels_controls):
    """Changing the iso gradient mode should update the layer and vice versa."""
    layer, qtctrl = make_labels_controls()
    qtctrl.ndisplay = 3
    assert layer.rendering == LabelsRendering.ISO_CATEGORICAL
    assert layer.iso_gradient_mode == IsoCategoricalGradientMode.FAST

    # Change the iso gradient mode via the control, check the layer
    qtctrl._render_control.iso_gradient_combobox.setCurrentEnum(
        IsoCategoricalGradientMode.SMOOTH
    )
    assert layer.iso_gradient_mode == IsoCategoricalGradientMode.SMOOTH

    # Change the iso gradient mode via the layer, check the control
    layer.iso_gradient_mode = IsoCategoricalGradientMode.FAST
    assert (
        qtctrl._render_control.iso_gradient_combobox.currentEnum()
        == IsoCategoricalGradientMode.FAST
    )


def test_iso_gradient_mode_hidden_for_2d(make_labels_controls):
    """Test that the iso gradient mode control is hidden with 2D view."""
    layer, qtctrl = make_labels_controls()
    assert qtctrl._render_control.iso_gradient_combobox.isHidden()
    layer.data = np.random.randint(5, size=(10, 15), dtype=np.uint8)
    assert qtctrl._render_control.iso_gradient_combobox.isHidden()
    qtctrl.ndisplay = 3
    assert not qtctrl._render_control.iso_gradient_combobox.isHidden()
    qtctrl.ndisplay = 2
    assert qtctrl._render_control.iso_gradient_combobox.isHidden()


def test_iso_gradient_mode_with_rendering(make_labels_controls):
    """Test the iso gradeint mode control is enabled for iso_categorical rendering."""
    layer, qtctrl = make_labels_controls()
    qtctrl.ndisplay = 3
    assert layer.rendering == LabelsRendering.ISO_CATEGORICAL
    assert (
        qtctrl._render_control.iso_gradient_combobox.currentText()
        == IsoCategoricalGradientMode.FAST
    )
    assert qtctrl._render_control.iso_gradient_combobox.isEnabled()
    layer.rendering = LabelsRendering.TRANSLUCENT
    assert not qtctrl._render_control.iso_gradient_combobox.isEnabled()
    layer.rendering = LabelsRendering.ISO_CATEGORICAL
    assert qtctrl._render_control.iso_gradient_combobox.isEnabled()
