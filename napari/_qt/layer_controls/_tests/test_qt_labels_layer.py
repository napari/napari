import numpy as np

from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.layers import Labels
from napari.utils.colormaps import colormap_utils

np.random.seed(0)
_LABELS = np.random.randint(5, size=(10, 15))
_COLOR = {1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'}


def test_changing_layer_color_mode_updates_combo_box(qtbot):
    """Updating layer color mode changes the combo box selection"""
    layer = Labels(_LABELS, color=_COLOR)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)

    original_color_mode = layer.color_mode
    assert original_color_mode == qtctrl.colorModeComboBox.currentText()

    layer.color_mode = 'auto'
    assert layer.color_mode == qtctrl.colorModeComboBox.currentText()


def test_rendering_combobox(qtbot):
    """Changing the model attribute should update the view"""
    layer = Labels(_LABELS)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)
    combo = qtctrl.renderComboBox
    opts = {combo.itemText(i) for i in range(combo.count())}
    rendering_options = {'translucent', 'iso_categorical'}
    assert opts == rendering_options
    # programmatically updating rendering mode updates the combobox
    new_mode = 'iso_categorical'
    layer.rendering = new_mode
    assert combo.findText(new_mode) == combo.currentIndex()


def test_changing_colormap_updates_colorbox(qtbot):
    """Test that changing the colormap on a layer will update color swatch in the combo box"""
    layer = Labels(_LABELS, color=_COLOR)
    qtctrl = QtLabelsControls(layer)
    qtbot.addWidget(qtctrl)
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
