import numpy as np
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIntValidator, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...utils.translations import translator

trans = translator.load()


class QtStar(QFrame):
    """Creates a star for the preview pane in the highlight widget.
    Parameters
    ----------
    value : int
        The line width of the star.
    """

    def __init__(
        self,
        parent,
        value: int = None,
    ):
        super().__init__(parent)
        # self.max_value = max_value
        # self.min_value = min_value
        self.value = value

    def sizeHint(self):
        """Override Qt sizeHint."""
        return QSize(100, 100)

    def minimumSizeHint(self):
        """Override Qt minimumSizeHint."""
        return QSize(100, 100)

    def paintEvent(self, e):
        """Paint star on frame."""
        qp = QPainter()
        qp.begin(self)

        self.drawStar(qp, self.value)
        qp.end()

    def getValue(self):
        """Return value of star widget."""
        return self.value

    def setValue(self, value):
        """Set value of star widget."""
        self.value = value
        self.update()

    def drawStar(self, qp, value):
        """Draw a star in the preview pane."""

        width = self.rect().width()
        height = self.rect().height()
        col = QColor(135, 206, 235)
        pen = QPen(col, value)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        qp.setPen(pen)

        path = QPainterPath()

        # draw pentagram
        star_center_x = width / 2
        star_center_y = height / 2
        # make sure the star equal no matter the size of the qframe
        if width < height:
            # not taking it all the way to the edge so the star has room to grow
            radius_outer = width * 0.35
        else:
            radius_outer = height * 0.35

        # start at the top point of the star and move counter clockwise to draw the path.
        # every other point is the shorter radius (1/(1+golden_ratio)) of the larger radius
        golden_ratio = (1 + np.sqrt(5)) / 2
        radius_inner = radius_outer / (1 + golden_ratio)
        theta_start = np.pi / 2
        theta_inc = (2 * np.pi) / 10
        for n in range(11):
            theta = theta_start + (n * theta_inc)
            theta = np.mod(theta, 2 * np.pi)
            if np.mod(n, 2) == 0:
                # use radius_outer
                x = radius_outer * np.cos(theta)
                y = radius_outer * np.sin(theta)

            else:
                # use radius_inner
                x = radius_inner * np.cos(theta)
                y = radius_inner * np.sin(theta)

            x_adj = star_center_x - x
            y_adj = star_center_y - y + 3
            if n == 0:

                path.moveTo(x_adj, y_adj)
            else:
                path.lineTo(x_adj, y_adj)

        qp.drawPath(path)


class QtTriangle(QFrame):
    """Draw the triangle in highlight widget.
    Parameters
    ----------
    value : int
        Current value of the highlight size.
    min_value : int
        Minimum value possible for highlight size.
    max_value : int
        Maximum value possible for highlight size.
    """

    valueChanged = Signal(int)

    def __init__(
        self,
        parent,
        value: int = None,
        min_value: int = 1,
        max_value: int = 10,
    ):
        super().__init__(parent)
        self.max_value = max_value
        self.min_value = min_value
        self.value = value

    def mousePressEvent(self, event):
        """When mouse is clicked, adjust to new values."""
        # set value based on position of event
        perc = event.pos().x() / self.rect().width()
        value = ((self.max_value - self.min_value) * perc) + self.min_value
        self.setValue(value)

    def paintEvent(self, e):
        """Paint triangle on frame."""
        qp = QPainter()
        qp.begin(self)
        self.drawTriangle(qp)
        perc = (self.value - self.min_value) / (
            self.max_value - self.min_value
        )
        self.drawLine(qp, self.rect().width() * perc)
        qp.end()

    def sizeHint(self):
        """Override Qt sizeHint."""
        return QSize(75, 30)

    def minimumSizeHint(self):
        """Override Qt minimumSizeHint."""
        return QSize(75, 30)

    def drawTriangle(self, qp):
        """Draw triangle."""
        width = self.rect().width()
        height = self.rect().height()

        # col = QColor('white')
        col = QColor(135, 206, 235)
        qp.setPen(QPen(col, 1))
        qp.setBrush(col)
        path = QPainterPath()

        height = 10
        # height = height * 0.85
        path.moveTo(0, height)

        path.lineTo(width, height)

        path.lineTo(width, 0)
        path.closeSubpath()

        qp.drawPath(path)

    def getValue(self):
        """Return value of triangle widget."""
        return self.value

    def setValue(self, value):
        """Set value for triangle widget."""
        self.value = value
        self.update()

    def drawLine(self, qp, value):
        """Draw line on triangle indicating value."""
        col = QColor('white')
        qp.setPen(QPen(col, 2))
        qp.setBrush(col)
        path = QPainterPath()
        path.moveTo(value, 15)

        path.lineTo(value, 0)
        path.closeSubpath()

        qp.drawPath(path)
        self.valueChanged.emit(self.value)


class HighlightSizePreviewWidget(QDialog):
    """Creates custom widget to set highlight size.
    Parameters
    ----------
    description : str
        Text to explain and display on widget.
    value : int
        Value of highlight size.
    min_value : int
        Minimum possible value of highlight size.
    max_value : int
        Maximum possible value of highlight size.
    unit : str
        Unit of highlight size.
    """

    valueChanged = Signal(int)

    def __init__(
        self,
        parent: QWidget = None,
        description: str = "",
        value: int = None,
        min_value: int = 1,
        max_value: int = 10,
        unit: str = "px",
    ):
        super().__init__(parent)

        self.setGeometry(300, 300, 125, 110)
        self._value = value if value else self.fontMetrics().height()
        self._min_value = min_value
        self._max_value = max_value

        # Widget
        self._lineedit = QLineEdit()
        self._description = QLabel(self)
        self._unit = QLabel(self)
        self._slider = QSlider(Qt.Horizontal)
        self._triangle = QtTriangle(
            self, value=value, min_value=min_value, max_value=max_value
        )
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        self._preview = QtStar(self, value)
        self._preview_label = QLabel(self)
        self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        self._description.setText(description)
        self._description.setWordWrap(True)
        self._unit.setText(unit)
        self._unit.setAlignment(Qt.AlignBottom)
        self._lineedit.setValidator(self._validator)
        self._lineedit.setAlignment(Qt.AlignRight)
        self._lineedit.setAlignment(Qt.AlignBottom)
        self._slider_min_label.setText(str(min_value))
        self._slider_min_label.setAlignment(Qt.AlignBottom)
        self._slider_max_label.setText(str(max_value))
        self._slider_max_label.setAlignment(Qt.AlignBottom)
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)
        self._preview_label.setText(trans._("Preview"))
        self._preview_label.setAlignment(Qt.AlignHCenter)
        self._preview_label.setAlignment(Qt.AlignBottom)
        self._preview.setStyleSheet('border: 1px solid white;')

        # Signals
        self._slider.valueChanged.connect(self.update_value)
        self._lineedit.textChanged.connect(self.update_value)
        self._triangle.valueChanged.connect(self.update_value)

        # Layout

        triangle_layout = QHBoxLayout()
        triangle_layout.addWidget(self._triangle)
        triangle_layout.setContentsMargins(6, 35, 6, 0)
        triangle_slider_layout = QVBoxLayout()
        triangle_slider_layout.addLayout(triangle_layout)
        triangle_slider_layout.setContentsMargins(0, 0, 0, 0)
        triangle_slider_layout.addWidget(self._slider)
        triangle_slider_layout.setAlignment(Qt.AlignVCenter)

        # Bottom row layout
        lineedit_layout = QHBoxLayout()
        lineedit_layout.addWidget(self._lineedit)
        lineedit_layout.setAlignment(Qt.AlignBottom)
        bottom_left_layout = QHBoxLayout()
        bottom_left_layout.addLayout(lineedit_layout)
        bottom_left_layout.addWidget(self._unit)
        bottom_left_layout.addWidget(self._slider_min_label)
        bottom_left_layout.addLayout(triangle_slider_layout)
        bottom_left_layout.addWidget(self._slider_max_label)
        bottom_left_layout.setAlignment(Qt.AlignBottom)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self._description)
        left_layout.addLayout(bottom_left_layout)
        left_layout.setAlignment(Qt.AlignLeft)

        preview_label_layout = QHBoxLayout()
        preview_label_layout.addWidget(self._preview_label)
        preview_label_layout.setAlignment(Qt.AlignHCenter)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self._preview)
        preview_layout.addLayout(preview_label_layout)
        preview_layout.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(preview_layout)

        self.setLayout(layout)

        self.refresh()

    def setMinimum(self, value):
        """Set slider mininum value."""
        self._slider.setMinimum(value)

    def update_value(self, value):
        """Update highlight value."""
        if value == "":
            value = int(self._value)

        self._value = int(value)
        self.refresh()

    def refresh(self):
        """Set every widget value to the new set value."""
        self.blockSignals(True)
        self._lineedit.setText(str(self._value))
        self._slider.setValue(self._value)
        self._triangle.setValue(self._value)
        self._preview.setValue(self._value)
        self.blockSignals(False)
        self.valueChanged.emit(self._value)

    def getValue(self):
        """Return current value."""
        return self._value

    def setValue(self, value):
        """Set new value and update widget."""
        self.update_value(value)
        self.refresh()

    def getDescription(self):
        """Return the description text."""
        return self._desctiption_label.text()

    def setDescription(self, text):
        """Set the description text."""
        self._desctiption_label.setText(text)

    def getPreviewText(self):
        """Return text for preview pane."""
        return self._preview.text()

    def setPreviewText(self, text):
        """Set text for preview pane."""
        self._preview.setText(text)

    def getUnit(self):
        """Return highlight value unit."""
        return self._unit_label.text()

    def setUnit(self, text):
        """Set highlight value unit."""
        self._unit_label.setText(text)

    def setMinimun(self, value):
        """Set minimum highlight value."""
        value = int(value)
        if value < self._max_value:
            self._min_value = value
            self._slider_min_label.setText(str(value))
            self._value = (
                self._min_value
                if self._value < self._min_value
                else self._value
            )
            self.refresh()
        else:
            raise ValueError(
                f"Minimum value must be smaller than {self._max_value}"
            )

    def getMinimun(self, value):
        """Return minimum highlight value."""
        return self._min_value

    def setMaximum(self, value):
        """Set maximum highlight value."""
        value = int(value)
        if value > self._max_value:
            self._max_value = value
            self._slider_max_label.setText(str(value))
            self._value = (
                self._max_value
                if self._value > self._max_value
                else self._value
            )
            self.refresh()
        else:
            raise ValueError(
                f"Maximum value must be larger than {self._min_value}"
            )

    def getMaximum(self, value):
        """Return maximum highlight value."""
        return self._max_value
