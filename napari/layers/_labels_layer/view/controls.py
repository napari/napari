from qtpy.QtGui import QPainter, QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QVBoxLayout,
    QRadioButton,
    QFrame,
    QWidget,
)
from .._constants import Mode
from ..._base_layer import QtLayerControls


class QtLabelsControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)

        self.panzoom_button = QtModeButton(
            layer, 'zoom', Mode.PAN_ZOOM, 'Pan/zoom mode'
        )
        self.pick_button = QtModeButton(
            layer, 'picker', Mode.PICKER, 'Pick mode'
        )
        self.paint_button = QtModeButton(
            layer, 'paint', Mode.PAINT, 'Paint mode'
        )
        self.fill_button = QtModeButton(layer, 'fill', Mode.FILL, 'Fill mode')

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.paint_button)
        self.button_group.addButton(self.pick_button)
        self.button_group.addButton(self.fill_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 20, 10, 10)
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.paint_button)
        layout.addWidget(self.pick_button)
        layout.addWidget(self.fill_button)
        layout.addWidget(QtColorBox(layer))
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.panzoom_button.setChecked(True)

    def mouseMoveEvent(self, event):
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.PICKER:
            self.pick_button.setChecked(True)
        elif mode == Mode.PAINT:
            self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            self.fill_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")


class QtModeButton(QRadioButton):
    def __init__(self, layer, button_name, mode, tool_tip):
        super().__init__()

        self.mode = mode
        self.layer = layer
        self.setToolTip(tool_tip)
        self.setChecked(False)
        self.setProperty('mode', button_name)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode


class QtColorBox(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self._height = 28
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip('Selected label color')

        self.layer.events.selected_label.connect(self.update_color)

    def update_color(self, event):
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self.layer._selected_color is None:
            painter.setPen(QColor(230, 230, 230))
            painter.setBrush(QColor(230, 230, 230))
            for i in range(self._height // 6 + 1):
                for j in range(self._height // 6 + 1):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.drawRect(i * 6, j * 6, 5, 5)
        else:
            color = 255 * self.layer._selected_color
            color = color.astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)
