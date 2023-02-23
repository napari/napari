import numpy as np
import pytest

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.utils.colormaps import colormap_utils

np.random.seed(0)
_LABELS = np.random.randint(5, size=(10, 15))
_COLOR = {1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'}


@pytest.fixture
def make_labels_controls(qtbot, color=None):
    def _make_labels_controls(color=color):
        layer = Labels(_LABELS, color=color)
        qtctrl = QtLabelsControls(layer)
        qtbot.add_widget(qtctrl)
        return layer, qtctrl

    return _make_labels_controls


def test_changing_layer_color_mode_updates_combo_box(make_labels_controls):
    """Updating layer color mode changes the combo box selection"""
    layer, qtctrl = make_labels_controls(color=_COLOR)

    original_color_mode = layer.color_mode
    assert original_color_mode == qtctrl.colorModeComboBox.currentText()

    layer.color_mode = 'auto'
    assert layer.color_mode == qtctrl.colorModeComboBox.currentText()


def test_rendering_combobox(make_labels_controls):
    """Changing the model attribute should update the view"""
    layer, qtctrl = make_labels_controls()
    combo = qtctrl.renderComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {'translucent', 'iso_categorical'}
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    new_mode = 'iso_categorical'
    layer.rendering = new_mode
    assert combo.findText(new_mode) == combo.currentIndex()


def test_changing_colormap_updates_colorbox(make_labels_controls):
    """Test that changing the colormap on a layer will update color swatch in the combo box"""
    layer, qtctrl = make_labels_controls(color=_COLOR)
    color_box = qtctrl.colorBox

    layer.selected_label = 1

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(
        color_box.color,
        np.round(np.asarray(layer._selected_color) * 255 * layer.opacity),
    )

    layer.colormap = colormap_utils.label_colormap(num_colors=5)

    # For a paint event, which does not occur in a headless qtbot
    color_box.paintEvent(None)

    np.testing.assert_equal(
        color_box.color,
        np.round(np.asarray(layer._selected_color) * 255 * layer.opacity),
    )


def test_selected_color_checkbox(make_labels_controls):
    """Tests that the 'selected color' checkbox sets the 'show_selected_label' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.selectedColorCheckbox.setChecked(True)
    assert layer.show_selected_label
    qtctrl.selectedColorCheckbox.setChecked(False)
    assert not layer.show_selected_label
    qtctrl.selectedColorCheckbox.setChecked(True)
    assert layer.show_selected_label


def test_contiguous_labels_checkbox(make_labels_controls):
    """Tests that the 'contiguous' checkbox sets the 'contiguous' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.contigCheckBox.setChecked(True)
    assert layer.contiguous
    qtctrl.contigCheckBox.setChecked(False)
    assert not layer.contiguous
    qtctrl.contigCheckBox.setChecked(True)
    assert layer.contiguous


def test_preserve_labels_checkbox(make_labels_controls):
    """Tests that the 'preserve labels' checkbox sets the 'preserve_labels' property properly."""
    layer, qtctrl = make_labels_controls()
    qtctrl.preserveLabelsCheckBox.setChecked(True)
    assert layer.preserve_labels
    qtctrl.preserveLabelsCheckBox.setChecked(False)
    assert not layer.preserve_labels
    qtctrl.preserveLabelsCheckBox.setChecked(True)
    assert layer.preserve_labels
