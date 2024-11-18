from typing import Optional

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
from napari.layers.base.base import Layer
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.events import disconnect_events
from napari.utils.translations import trans


class QtColorBox(QWidget):
    """A widget that shows a square with the current label color.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari layer.
    """

    def __init__(self, layer) -> None:
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

    def _on_selected_label_change(self):
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_opacity_change(self):
        """Receive layer model label selection change event & update colorbox."""
        self.update()

    def _on_colormap_change(self):
        """Receive label colormap change event & update colorbox."""
        self.update()

    def paintEvent(self, event):
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

    def deleteLater(self):
        disconnect_events(self._layer.events, self)
        super().deleteLater()

    def closeEvent(self, event):
        """Disconnect events when widget is closing."""
        disconnect_events(self._layer.events, self)
        super().closeEvent(event)


class QtLabelControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the current label
    layer attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
        labelColor : qtpy.QtWidget.QWidget
            Wrapper widget for the selectionSpinBox and colorBox widgets.
        selectionSpinBox : superqt.QLargeIntSpinBox
            Widget to select a specific label by its index.
            N.B. cannot represent labels > 2**53.
        colorBox: QtColorBox
            Widget that shows current layer label color.
        labelColorLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
            Label for the label chooser widget.
    """

    def __init__(
        self, parent: QWidget, layer: Layer, tooltip: Optional[str] = None
    ) -> None:
        super().__init__(parent, layer)
        # Setup layer
        self._layer.events.selected_label.connect(
            self._on_selected_label_change
        )
        self._layer.events.data.connect(self._on_data_change)

        # Setup widgets
        self.selectionSpinBox = QLargeIntSpinBox()
        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.selectionSpinBox.setRange(*dtype_lims)
        self.selectionSpinBox.setKeyboardTracking(False)
        self.selectionSpinBox.valueChanged.connect(self.changeSelection)
        self.selectionSpinBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._on_selected_label_change()

        self.labelColorLabel = QtWrappedLabel(trans._('label:'))
        self.labelColor = QWidget()
        self.labelColor.setProperty('emphasized', True)
        color_layout = QHBoxLayout()
        self.colorBox = QtColorBox(layer)
        color_layout.addWidget(self.colorBox)
        color_layout.addWidget(self.selectionSpinBox)
        self.labelColor.setLayout(color_layout)

    def changeSelection(self, value):
        """Change currently selected label.

        Parameters
        ----------
        value : int
            Index of label to select.
        """
        self._layer.selected_label = value
        self.selectionSpinBox.clearFocus()
        # TODO: decouple
        self.parent().setFocus()

    def _on_selected_label_change(self):
        """Receive layer model label selection change event and update spinbox."""
        with self._layer.events.selected_label.blocker():
            value = self._layer.selected_label
            self.selectionSpinBox.setValue(value)

    def _on_data_change(self):
        """Update label selection spinbox min/max when data changes."""
        dtype_lims = get_dtype_limits(get_dtype(self._layer))
        self.selectionSpinBox.setRange(*dtype_lims)

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.labelColorLabel, self.labelColor)]