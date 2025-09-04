import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QHBoxLayout,
    QWidget,
)
from superqt import QLargeIntSpinBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import qt_signals_blocked
from napari.layers import Labels
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.events import disconnect_events
from napari.utils.translations import trans


class QtColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.
    """

    def __init__(self, layer: Labels) -> None:
        super().__init__()

        self._layer = layer
        self._layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self._layer.events.opacity.connect(self._on_opacity_change)
        self._layer.events.colormap.connect(self._on_colormap_change)

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._height = 24
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip(trans._('Selected label color'))

        self.color = None

    def _on_selected_label_change(self) -> None:
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_opacity_change(self) -> None:
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_colormap_change(self) -> None:
        """Receive label colormap change event & update colorbox."""
        self.update()

    def paintEvent(self, event) -> None:
        """Paint the colorbox.  If no color, display a checkerboard pattern.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self._layer._selected_color is None:
            self.color = None
            for i in range(self._height // 4):
                for j in range(self._height // 4):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.setPen(QColor(230, 230, 230))
                        painter.setBrush(QColor(230, 230, 230))
                    else:
                        painter.setPen(QColor(25, 25, 25))
                        painter.setBrush(QColor(25, 25, 25))
                    painter.drawRect(i * 4, j * 4, 5, 5)
        else:
            color = np.round(255 * self._layer._selected_color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)
            self.color = tuple(color)

    def disconnect_widget_controls(self) -> None:
        disconnect_events(self._layer.events, self)

    def deleteLater(self) -> None:
        self.disconnect_widget_controls()
        super().deleteLater()

    def closeEvent(self, event) -> None:
        """Disconnect events when widget is closing."""
        self.disconnect_widget_controls()
        super().closeEvent(event)


class QtLabelControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current label
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    colorbox : QtColorBox
        Widget that shows current layer label color.
    label_color : qtpy.QtWidget.QWidget
        Wrapper widget for the selection_spinbox and colorbox widgets.
    label_color_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the label chooser widget.
    selection_spinbox : superqt.QLargeIntSpinBox
        Widget to select a specific label by its index.
        N.B. cannot represent labels > 2**53.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self._layer.events.data.connect(self._on_data_change)

        # Setup widgets
        self.selection_spinbox = QLargeIntSpinBox()
        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.selection_spinbox.setRange(*dtype_lims)
        self.selection_spinbox.setKeyboardTracking(False)
        self.selection_spinbox.valueChanged.connect(self.change_selection)
        self.selection_spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_selected_label_change()

        self.label_color_label = QtWrappedLabel(trans._('label:'))
        self.label_color = QWidget()
        self.label_color.setProperty('emphasized', True)
        color_layout = QHBoxLayout()
        color_layout.setContentsMargins(0, 2, 0, 1)
        self.colorbox = QtColorBox(layer)
        color_layout.addWidget(self.colorbox)
        color_layout.addWidget(self.selection_spinbox)
        self.label_color.setLayout(color_layout)

    def change_selection(self, value: int) -> None:
        """Change currently selected label.

        Parameters
        ----------
        value : int
            Index of label to select.
        """
        self._layer.selected_label = value
        self.selection_spinbox.clearFocus()
        # TODO: decouple
        self.parent().setFocus()

    def _on_selected_label_change(self) -> None:
        """Receive layer model label selection change event and update spinbox."""
        with qt_signals_blocked(self.selection_spinbox):
            value = self._layer.selected_label
            self.selection_spinbox.setValue(value)

    def _on_data_change(self) -> None:
        """Update label selection spinbox min/max when data changes."""
        with qt_signals_blocked(self.selection_spinbox):
            dtype_lims = get_dtype_limits(get_dtype(self._layer))
            self.selection_spinbox.setRange(*dtype_lims)

    def disconnect_widget_controls(self) -> None:
        self.colorbox.disconnect_widget_controls()
        super().disconnect_widget_controls()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.label_color_label, self.label_color)]
