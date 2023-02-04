import numpy as np
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIntValidator, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from napari.utils.translations import translator

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
        parent: QWidget = None,
        value: int = None,
    ) -> None:
        super().__init__(parent)
        self._value = value

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

        self.drawStar(qp)
        qp.end()

    def value(self):
        """Return value of star widget.

        Returns
        -------
        int
            The value of the star widget.

        """
        return self._value

    def setValue(self, value: int):
        """Set line width value of star widget.

        Parameters
        ----------
        value : int
            line width value for star
        """

        self._value = value
        self.update()

    def drawStar(self, qp):
        """Draw a star in the preview pane.

        Parameters
        ----------
        qp : QPainter object
        """
        width = self.rect().width()
        height = self.rect().height()
        col = QColor(135, 206, 235)
        pen = QPen(col, self._value)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        qp.setPen(pen)

        path = QPainterPath()

        # draw pentagram
        star_center_x = width / 2
        star_center_y = height / 2
        # make sure the star equal no matter the size of the qframe
        radius_outer = width * 0.35 if width < height else height * 0.35
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
        parent: QWidget = None,
        value: int = 1,
        min_value: int = 1,
        max_value: int = 10,
    ) -> None:
        super().__init__(parent)
        self._max_value = max_value
        self._min_value = min_value
        self._value = value

    def mousePressEvent(self, event):
        """When mouse is clicked, adjust to new values."""
        # set value based on position of event
        perc = event.pos().x() / self.rect().width()
        value = ((self._max_value - self._min_value) * perc) + self._min_value
        self.setValue(value)

    def paintEvent(self, e):
        """Paint triangle on frame."""
        qp = QPainter()
        qp.begin(self)
        self.drawTriangle(qp)
        perc = (self._value - self._min_value) / (
            self._max_value - self._min_value
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
        """Draw triangle.

        Parameters
        ----------
        qp : QPainter object
        """
        width = self.rect().width()

        col = QColor(135, 206, 235)
        qp.setPen(QPen(col, 1))
        qp.setBrush(col)
        path = QPainterPath()

        height = 10
        path.moveTo(0, height)

        path.lineTo(width, height)

        path.lineTo(width, 0)
        path.closeSubpath()

        qp.drawPath(path)

    def value(self):
        """Return value of triangle widget.

        Returns
        -------
        int
            Current value of triangle widget.
        """
        return self._value

    def setValue(self, value):
        """Set value for triangle widget.

        Parameters
        ----------
        value : int
            Value to use for line in triangle widget.
        """
        self._value = value
        self.update()

    def minimum(self):
        """Return minimum value.

        Returns
        -------
        int
            Mininum value of triangle widget.
        """
        return self._min_value

    def maximum(self):
        """Return maximum value.

        Returns
        -------
        int
            Maximum value of triangle widget.
        """
        return self._max_value

    def setMinimum(self, value: int):
        """Set minimum value

        Parameters
        ----------
        value : int
            Minimum value of triangle.
        """
        self._min_value = value
        self._value = max(self._value, value)

    def setMaximum(self, value: int):
        """Set maximum value.

        Parameters
        ----------
        value : int
            Maximum value of triangle.
        """
        self._max_value = value

        self._value = min(self._value, value)

    def drawLine(self, qp, value: int):
        """Draw line on triangle indicating value.

        Parameters
        ----------
        qp : QPainter object
        value : int
            Value of highlight thickness.
        """
        col = QColor('white')
        qp.setPen(QPen(col, 2))
        qp.setBrush(col)
        path = QPainterPath()
        path.moveTo(value, 15)

        path.lineTo(value, 0)
        path.closeSubpath()

        qp.drawPath(path)
        self.valueChanged.emit(self._value)


class QtHighlightSizePreviewWidget(QWidget):
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
        value: int = 1,
        min_value: int = 1,
        max_value: int = 10,
        unit: str = "px",
    ) -> None:
        super().__init__(parent)

        self.setGeometry(300, 300, 125, 110)
        self._value = value or self.fontMetrics().height()
        self._min_value = min_value
        self._max_value = max_value

        # Widget
        self._lineedit = QLineEdit()
        self._description = QLabel(self)
        self._unit = QLabel(self)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._triangle = QtTriangle(self)
        self._slider_min_label = QLabel(self)
        self._slider_max_label = QLabel(self)
        self._preview = QtStar(self)
        self._preview_label = QLabel(self)
        self._validator = QIntValidator(min_value, max_value, self)

        # Widgets setup
        self._description.setText(description)
        self._description.setWordWrap(True)
        self._unit.setText(unit)
        self._unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._lineedit.setValidator(self._validator)
        self._lineedit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lineedit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._slider_min_label.setText(str(min_value))
        self._slider_min_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._slider_max_label.setText(str(max_value))
        self._slider_max_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)
        self._preview.setValue(value)
        self._triangle.setValue(value)
        self._triangle.setMinimum(min_value)
        self._triangle.setMaximum(max_value)
        self._preview_label.setText(trans._("Preview"))
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self._preview.setStyleSheet('border: 1px solid white;')

        # Signals
        self._slider.valueChanged.connect(self._update_value)
        self._lineedit.textChanged.connect(self._update_value)
        self._triangle.valueChanged.connect(self._update_value)

        # Layout
        triangle_layout = QHBoxLayout()
        triangle_layout.addWidget(self._triangle)
        triangle_layout.setContentsMargins(6, 35, 6, 0)
        triangle_slider_layout = QVBoxLayout()
        triangle_slider_layout.addLayout(triangle_layout)
        triangle_slider_layout.setContentsMargins(0, 0, 0, 0)
        triangle_slider_layout.addWidget(self._slider)
        triangle_slider_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Bottom row layout
        lineedit_layout = QHBoxLayout()
        lineedit_layout.addWidget(self._lineedit)
        lineedit_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        bottom_left_layout = QHBoxLayout()
        bottom_left_layout.addLayout(lineedit_layout)
        bottom_left_layout.addWidget(self._unit)
        bottom_left_layout.addWidget(self._slider_min_label)
        bottom_left_layout.addLayout(triangle_slider_layout)
        bottom_left_layout.addWidget(self._slider_max_label)
        bottom_left_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self._description)
        left_layout.addLayout(bottom_left_layout)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        preview_label_layout = QHBoxLayout()
        preview_label_layout.addWidget(self._preview_label)
        preview_label_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self._preview)
        preview_layout.addLayout(preview_label_layout)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(preview_layout)

        self.setLayout(layout)

        self._refresh()

    def _update_value(self, value):
        """Update highlight value.

        Parameters
        ----------
        value : int
            Highlight value.
        """
        if value == "":
            return
        value = int(value)
        value = max(min(value, self._max_value), self._min_value)
        if value == self._value:
            return
        self._value = value
        self.valueChanged.emit(self._value)
        self._refresh()

    def _refresh(self):
        """Set every widget value to the new set value."""
        self.blockSignals(True)
        self._lineedit.setText(str(self._value))
        self._slider.setValue(self._value)
        self._triangle.setValue(self._value)
        self._preview.setValue(self._value)
        self.blockSignals(False)

    def value(self):
        """Return current value.

        Returns
        -------
        int
            Current value of highlight widget.
        """
        return self._value

    def setValue(self, value):
        """Set new value and update widget.

        Parameters
        ----------
        value : int
            Highlight value.
        """
        self._update_value(value)
        self._refresh()

    def description(self):
        """Return the description text.

        Returns
        -------
        str
            Current text in description.
        """
        return self._description.text()

    def setDescription(self, text):
        """Set the description text.

        Parameters
        ----------
        text : str
            Text to use in description box.
        """
        self._description.setText(text)

    def unit(self):
        """Return highlight value unit text.

        Returns
        -------
        str
            Current text in unit text.
        """
        return self._unit.text()

    def setUnit(self, text):
        """Set highlight value unit.

        Parameters
        ----------
        text : str
            Text used to describe units.
        """
        self._unit.setText(text)

    def setMinimum(self, value):
        """Set minimum highlight value for star, triangle, text and slider.

        Parameters
        ----------
        value : int
            Minimum highlight value.
        """
        value = int(value)
        if value >= self._max_value:
            raise ValueError(
                trans._(
                    "Minimum value must be smaller than {max_value}",
                    deferred=True,
                    max_value=self._max_value,
                )
            )
        self._min_value = value
        self._slider_min_label.setText(str(value))
        self._slider.setMinimum(value)
        self._triangle.setMinimum(value)
        self._value = max(self._value, self._min_value)
        self._refresh()

    def minimum(self):
        """Return minimum highlight value.

        Returns
        -------
        int
            Minimum value of highlight widget.
        """
        return self._min_value

    def setMaximum(self, value):
        """Set maximum highlight value.

        Parameters
        ----------
        value : int
            Maximum highlight value.
        """
        value = int(value)
        if value <= self._min_value:
            raise ValueError(
                trans._(
                    "Maximum value must be larger than {min_value}",
                    deferred=True,
                    min_value=self._min_value,
                )
            )
        self._max_value = value
        self._slider_max_label.setText(str(value))
        self._slider.setMaximum(value)
        self._triangle.setMaximum(value)
        self._value = min(self._value, self._max_value)
        self._refresh()

    def maximum(self):
        """Return maximum highlight value.

        Returns
        -------
        int
            Maximum value of highlight widget.
        """
        return self._max_value
