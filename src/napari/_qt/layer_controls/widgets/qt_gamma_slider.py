from qtpy.QtCore import Qt
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari._qt.widgets.qt_mode_buttons import QtModeRadioButton
from napari.layers.base.base import Layer
from napari.utils.events.event_utils import connect_setattr
from napari.utils.translations import trans


class QtGammaSliderControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current gamma
    attribute value and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    gamma_slider : superqt.QLabeledDoubleSlider
        Gamma adjustment slider widget.
    gamma_slider_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the gamma chooser widget.
    histogram_button : QtModeRadioButton
        Button to toggle histogram widget.
    histogram_visible : bool
        Whether the histogram widget is currently visible.
    """

    def __init__(self, parent, layer: Layer) -> None:
        super().__init__(parent, layer)

        # Track histogram visibility state
        self.histogram_visible = False

        # Setup gamma slider - exactly like opacity slider
        sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=parent)
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(0.2)
        sld.setMaximum(2)
        sld.setSingleStep(0.02)
        sld.setValue(self._layer.gamma)
        connect_setattr(sld.valueChanged, self._layer, 'gamma')
        self._callbacks.append(
            attr_to_settr(self._layer, 'gamma', sld, 'setValue')
        )
        self.gamma_slider = sld
        self.gamma_slider_label = QtWrappedLabel(trans._('gamma:'))

        # Create histogram button on same row by appending to slider's layout
        self.histogram_button = QtModeRadioButton(
            layer,
            button_name='histogram',
            mode=None,
            slot=self._on_histogram_button_clicked,
            tooltip=(
                'Left click to toggle histogram in layer controls.\n'
                'Right click to open histogram popup.'
            ),
        )
        # Install event filter for right-click handling
        self.histogram_button.installEventFilter(self)

        # Add button directly to slider's layout
        sld.layout().addWidget(self.histogram_button)

    def eventFilter(self, obj, event):
        """Handle right-click on histogram button to show popup.

        Parameters
        ----------
        obj : QObject
            The object that generated the event.
        event : QEvent
            The event to handle.

        Returns
        -------
        bool
            True if the event was handled, False otherwise.
        """
        if (
            obj == self.histogram_button
            and event.type() == event.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.show_histogram_popup()
            return True
        return super().eventFilter(obj, event)

    def _on_histogram_button_clicked(self):
        """Handle left-click on histogram button to toggle histogram widget."""
        parent = self.parent()
        if not hasattr(parent, '_histogram_control'):
            return

        # Get the layout (QFormLayout)
        layout = parent.layout()

        if self.histogram_visible:
            # Remove histogram widget from layout
            label_item = layout.labelForField(
                parent._histogram_control.content_widget
            )
            if label_item is not None:
                layout.removeWidget(label_item)
                label_item.hide()
            layout.removeWidget(parent._histogram_control.content_widget)
            parent._histogram_control.content_widget.hide()

            # Disable histogram computation
            self._layer.histogram.enabled = False

            self.histogram_visible = False
            self.histogram_button.setChecked(False)
        else:
            # Add histogram widget to layout (after gamma slider)
            # Find the row index of the gamma slider
            gamma_row = -1
            for i in range(layout.rowCount()):
                if layout.itemAt(i, layout.ItemRole.FieldRole):
                    widget = layout.itemAt(
                        i, layout.ItemRole.FieldRole
                    ).widget()
                    if widget == self.gamma_slider:
                        gamma_row = i
                        break

            if gamma_row >= 0:
                # Insert histogram widget right after gamma slider
                histogram_label = QtWrappedLabel(trans._('histogram:'))
                layout.insertRow(
                    gamma_row + 1,
                    histogram_label,
                    parent._histogram_control.content_widget,
                )
                parent._histogram_control.content_widget.show()
                # Enable histogram computation and force update
                self._layer.histogram.enabled = True
                self._layer.histogram.compute()
                self.histogram_visible = True
                self.histogram_button.setChecked(True)

    def show_histogram_popup(self):
        """Show the histogram popup widget."""
        # Enable histogram if not already enabled
        if not self._layer.histogram.enabled:
            self._layer.histogram.enabled = True
            self._layer.histogram.compute()

        # Access the parent's contrast limits control to show the popup
        parent = self.parent()
        if hasattr(parent, '_contrast_limits_control'):
            parent._contrast_limits_control.show_clim_popup()

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QLabeledDoubleSlider]]:
        return [(self.gamma_slider_label, self.gamma_slider)]
