from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton
from superqt import QLabeledDoubleSlider

from napari._qt.layer_controls.widgets.qt_histogram_control import (
    layer_supports_histogram_ui,
)
from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
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
    histogram_button : QPushButton
        Button to toggle histogram widget.
    """

    def __init__(self, parent, layer: Layer) -> None:
        super().__init__(parent, layer)

        self.histogram_button = None

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

        if layer_supports_histogram_ui(layer):
            self.histogram_button = QPushButton(parent)
            self.histogram_button.setProperty('mode', 'histogram')
            self.histogram_button.setToolTip(
                'Left click to toggle histogram in layer controls.\n'
                'Right click to open histogram popup.'
            )
            self.histogram_button.setCheckable(True)
            self.histogram_button.setFixedSize(28, 28)
            self.histogram_button.toggled.connect(
                self._on_histogram_button_toggled
            )
            self.histogram_button.installEventFilter(self)
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
            self.histogram_button is not None
            and obj == self.histogram_button
            and event.type() == event.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            self.histogram_button.setDown(False)
            self.show_histogram_popup()
            return True
        return super().eventFilter(obj, event)

    def _on_histogram_button_toggled(self, visible: bool) -> None:
        """Handle left-click on histogram button to toggle histogram widget."""
        if not layer_supports_histogram_ui(self._layer):
            return

        parent = self.parent()
        histogram_control = getattr(parent, '_histogram_control', None)
        if histogram_control is None:
            return

        histogram_control.ensure_content()

        # Get the layout (QFormLayout)
        layout = parent.layout()

        if not visible:
            # Remove histogram widget from layout
            layout.removeWidget(histogram_control.content_widget)
            histogram_control.content_widget.hide()

            # Disable histogram computation
            self._layer.histogram.enabled = False
        else:
            # Add histogram widget to layout (after gamma slider)
            gamma_row = -1
            for row in range(layout.rowCount()):
                item = layout.itemAt(row, layout.ItemRole.FieldRole)
                if item and item.widget() is self.gamma_slider:
                    gamma_row = row
                    break

            if gamma_row >= 0:
                # Span the histogram across the full form width.
                layout.insertRow(
                    gamma_row + 1, histogram_control.content_widget
                )
                histogram_control.content_widget.show()
                # Enable histogram computation and force update
                self._layer.histogram.enabled = True
                self._layer.histogram.compute()

    def show_histogram_popup(self):
        """Show the histogram popup widget."""
        if not layer_supports_histogram_ui(self._layer):
            return

        # The popup's showEvent manages histogram enable/disable; do not
        # pre-enable here, or the popup cannot tell whether it was the one
        # that enabled it and will skip the matching disable on close.
        self.parent()._contrast_limits_control.show_clim_popup()

    def get_widget_controls(
        self,
    ) -> list[tuple[QtWrappedLabel, QLabeledDoubleSlider]]:
        return [(self.gamma_slider_label, self.gamma_slider)]
