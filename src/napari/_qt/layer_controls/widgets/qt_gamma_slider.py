from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QWidget
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
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
    histogram_visible : bool
        Whether the histogram widget is currently visible.
    """

    def __init__(self, parent: QWidget, layer: Layer) -> None:
        super().__init__(parent, layer)

        # Track histogram visibility state
        self.histogram_visible = False

        # Create slider container with button
        self.slider_container = QWidget(parent)
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        # Setup gamma slider - use parent as parent for proper QSS inheritance
        sld = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal,
            parent=parent
        )
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

        # Create histogram button - pass layer as first argument
        self.histogram_button = QtModePushButton(
            layer,
            button_name='histogram',
            slot=self._on_histogram_button_clicked,
            tooltip=(
                'Left click to toggle histogram in layer controls.\n'
                'Right click to open histogram popup.'
            )
        )
        self.histogram_button.setCheckable(True)
        # Install event filter for right-click handling
        self.histogram_button.installEventFilter(self)

        # Add widgets to container layout
        container_layout.addWidget(sld)
        container_layout.addWidget(self.histogram_button)
        self.slider_container.setLayout(container_layout)

        self.gamma_slider_label = QtWrappedLabel(trans._('gamma:'))

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
            label_item = layout.labelForField(parent._histogram_control.content_widget)
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
                    widget = layout.itemAt(i, layout.ItemRole.FieldRole).widget()
                    if widget == self.slider_container:
                        gamma_row = i
                        break
            
            if gamma_row >= 0:
                # Insert histogram widget right after gamma slider
                histogram_label = QtWrappedLabel(trans._('histogram:'))
                layout.insertRow(
                    gamma_row + 1,
                    histogram_label,
                    parent._histogram_control.content_widget
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

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.gamma_slider_label, self.slider_container)]
