from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets._labels import (
    QtBrushSizeSliderControl,
    QtColorModeComboBoxControl,
    QtContiguousCheckBoxControl,
    QtContourSpinBoxControl,
    QtDisplaySelectedLabelCheckBoxControl,
    QtLabelControl,
    QtLabelRenderControl,
    QtNdimSpinBoxControl,
    QtPreserveLabelsCheckBoxControl,
)
from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.labels._labels_constants import Mode
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


INT32_MAX = 2**31 - 1


class QtLabelsControls(QtLayerControls):
    """Qt view and controls for the napari Labels layer.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    _brush_size_slider_control : napari._qt.layer_controls.widgets._labels.QtBrushSizeSliderControl
        Widget that wraps the slider controlling current layer brush size.
    _colormode_combobox_control : napari._qt.layer_controls.widgets._labels.QtColorModeComboBoxControl
        Widget that wraps the comboBox controlling current color mode of the layer.
    _contiguous_checkbox_control : napari._qt.layer_controls.widgets._labels.QtContiguousCheckBoxControl
        Widget that wraps the checkbox to control if label layer fill in is contiguous.
    _contour_spinbox_control : napari._qt.layer_controls.widgets._labels.QtContourSpinBoxControl
        Widget that wraps the spinbox to control the layer contour thickness.
    _display_selected_label_checkbox_control : napari._qt.layer_controls.widgets._labels.QtDisplaySelectedLabelCheckBoxControl
        Widget that wraps the checkbox to control if only the current selected label is shown.
    _ndim_spinbox_control : napari._qt.layer_controls.widgets._labels.QtNdimSpinBoxControl
        Widget that wraps the spinbox to control the number of editable layer dimensions.
    _label_control : napari._qt.layer_controls.widgets._labels.QtLabelControl
        Wrapper widget to handle current label selection.
    _preserve_labels_checkbox_control :napari._qt.layer_controls.widgets._labels.QtPreserveLabelsCheckBoxControl
        Widget that wraps the checkbox to control if existing labels are preserved.
    _render_control : napari._qt.layer_controls.widgets._labels.QtLabelRenderControl
        Wrapper widget to control current label render method.
    colormap_update : napari._qt.widgets.qt_mode_buttons.QtModePushButton
        Button to update colormap of label layer.
    erase_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select ERASE mode on Labels layer.
    fill_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select FILL mode on Labels layer.
    paint_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PAINT mode on Labels layer.
    pick_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PICK mode on Labels layer.
    polygon_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select POLYGON mode on Labels layer.

    Raises
    ------
    ValueError
        Raise error if label mode is not PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    """

    layer: 'napari.layers.Labels'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_labels_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_labels_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        # shuffle colormap button
        self.colormap_update = QtModePushButton(
            layer,
            'shuffle',
            slot=self.changeColor,
            tooltip=trans._('Shuffle colors'),
        )

        self.pick_button = self._radio_button(
            layer,
            'picker',
            Mode.PICK,
            True,
            'activate_labels_picker_mode',
        )
        self.paint_button = self._radio_button(
            layer,
            'paint',
            Mode.PAINT,
            True,
            'activate_labels_paint_mode',
        )
        self.polygon_button = self._radio_button(
            layer,
            'labels_polygon',
            Mode.POLYGON,
            True,
            'activate_labels_polygon_mode',
        )
        self.fill_button = self._radio_button(
            layer,
            'fill',
            Mode.FILL,
            True,
            'activate_labels_fill_mode',
        )
        self.erase_button = self._radio_button(
            layer,
            'erase',
            Mode.ERASE,
            True,
            'activate_labels_erase_mode',
        )
        # don't bind with action manager as this would remove "Toggle with {shortcut}"
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.colormap_update, 0, 0)
        self.button_grid.addWidget(self.erase_button, 0, 1)
        self.button_grid.addWidget(self.paint_button, 0, 2)
        self.button_grid.addWidget(self.polygon_button, 0, 3)
        self.button_grid.addWidget(self.fill_button, 0, 4)
        self.button_grid.addWidget(self.pick_button, 0, 5)

        # Setup widgets controls
        self._label_control = QtLabelControl(self, layer)
        self._add_widget_controls(self._label_control)
        self._brush_size_slider_control = QtBrushSizeSliderControl(self, layer)
        self._add_widget_controls(self._brush_size_slider_control)
        self._render_control = QtLabelRenderControl(self, layer)
        self._add_widget_controls(self._render_control)
        self._colormode_combobox_control = QtColorModeComboBoxControl(
            self, layer
        )
        self._add_widget_controls(self._colormode_combobox_control)
        self._contour_spinbox_control = QtContourSpinBoxControl(self, layer)
        self._add_widget_controls(self._contour_spinbox_control)
        self._ndim_spinbox_control = QtNdimSpinBoxControl(self, layer)
        self._add_widget_controls(self._ndim_spinbox_control)
        self._contiguous_checkbox_control = QtContiguousCheckBoxControl(
            self, layer
        )
        self._add_widget_controls(self._contiguous_checkbox_control)
        self._preserve_labels_checkbox_control = (
            QtPreserveLabelsCheckBoxControl(self, layer)
        )
        self._add_widget_controls(self._preserve_labels_checkbox_control)
        self._display_selected_label_checkbox_control = (
            QtDisplaySelectedLabelCheckBoxControl(self, layer)
        )
        self._add_widget_controls(
            self._display_selected_label_checkbox_control
        )

        self._on_ndisplay_changed()

    def _on_mode_change(self, event):
        """Receive layer model mode change event and update checkbox ticks.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM, PICK, PAINT, ERASE, FILL
            or TRANSFORM
        """
        super()._on_mode_change(event)

    def changeColor(self):
        """Change colormap of the label layer."""
        self.layer.new_colormap()

    def _on_editable_or_visible_change(self):
        super()._on_editable_or_visible_change()
        self._set_polygon_tool_state()

    def _on_ndisplay_changed(self):
        show_3d_widgets = self.ndisplay == 3
        if show_3d_widgets:
            self._render_control._on_display_change_show()
        else:
            self._render_control._on_display_change_hide()
        self._on_editable_or_visible_change()
        self._set_polygon_tool_state()
        super()._on_ndisplay_changed()

    def _set_polygon_tool_state(self):
        if hasattr(self, 'polygon_button'):
            set_widgets_enabled_with_opacity(
                self, [self.polygon_button], self._is_polygon_tool_enabled()
            )

    def _is_polygon_tool_enabled(self):
        return (
            self.layer.editable
            and self.layer.visible
            and self.layer.n_edit_dimensions == 2
            and self.ndisplay == 2
        )
