"""
Range slider, extended QWidget slider for napari.
"""

from qtpy.QtGui import QPainter, QColor
from qtpy.QtCore import Signal, Property, Qt
from qtpy.QtWidgets import QWidget
from ..utils.validators import validate_n_seq


validate_2_tuple = validate_n_seq(2)


class QRangeSlider(QWidget):
    """
    QRangeSlider class, super class for QVRangeSlider and QHRangeSlider.
    """

    valuesChanged = Signal(tuple)
    rangeChanged = Signal(tuple)
    collapsedChanged = Signal(bool)
    focused = Signal()
    resized = Signal()

    def __init__(
        self,
        initial_values=None,
        data_range=None,
        step_size=None,
        collapsible=True,
        collapsed=False,
        parent=None,
    ):
        """A range slider with two handles for min/max values.

        Values should be provided in the range of the underlying data.
        (normalization to 0-1 happens internally in the slider.sliderValues())

        Parameters
        ----------
        initial_values : 2-tuple, optional
            Initial min & max values of the slider, defaults to (0.2, 0.8)
        data_range : 2-tuple, optional
            Min and max of the slider range, defaults to (0, 1)
        step_size : float, optional
            Single step size for the slider, defaults to 1
        collapsible : bool
            Whether the slider is collapsible, defaults to True.
        collapsed : bool
            Whether the slider begins collapsed, defaults to False.
        parent : qtpy.QtWidgets.QWidget
            Parent widget.
        """
        super().__init__(parent)
        self.handle_radius = 8
        self.slider_width = 8
        self.moving = "none"
        self.collapsible = collapsible
        self.collapsed = collapsed
        self.prev_moving = None
        self.bc_min = None
        self.bc_max = None

        # Variables initialized in methods
        self.value_min = 0
        self.value_max = 1
        self.start_display_min = None
        self.start_display_max = None
        self.start_pos = None
        self.display_min = None
        self.display_max = None

        self.setBarColor(QColor(200, 200, 200))
        self.setBackgroundColor(QColor(100, 100, 100))
        self.setHandleColor(QColor(200, 200, 200))
        self.setHandleBorderColor(QColor(200, 200, 200))

        self.setRange((0, 100) if data_range is None else data_range)
        self.setValues((20, 80) if initial_values is None else initial_values)
        if step_size is None:
            # pick an appropriate slider step size based on the data range
            if data_range is not None:
                step_size = (data_range[1] - data_range[0]) / 1000
            else:
                step_size = 0.001
        self.setStep(step_size)
        if not parent:
            if 'HRange' in self.__class__.__name__:
                self.setGeometry(200, 200, 200, 20)
            else:
                self.setGeometry(200, 200, 20, 200)

    def range(self):
        """Min and max possible values for the slider range. In data units"""
        return self.data_range_min, self.data_range_max

    def setRange(self, values):
        """Min and max possible values for the slider range. In data units."""
        validate_2_tuple(values)
        self.data_range_min, self.data_range_max = values
        self.rangeChanged.emit(self.range())
        self.updateDisplayPositions()

    def values(self):
        """Current slider values.

        Returns
        -------
        tuple
            Current minimum and maximum values of the range slider
        """
        return tuple(
            [self._slider_to_data_value(v) for v in self.sliderValues()]
        )

    def setValues(self, values):
        self.setSliderValues([self._data_to_slider_value(v) for v in values])

    def sliderValues(self):
        """Current slider values, as a fraction of slider width.

        Returns
        -------
        values : 2-tuple of int
            Start and end of the range.
        """
        return self.value_min, self.value_max

    def setSliderValues(self, values):
        """Set current slider values, as a fraction of slider width.

        Parameters
        ----------
        values : 2-tuple of float or int
            Start and end of the range.
        """
        validate_2_tuple(values)
        self.value_min, self.value_max = values
        self.valuesChanged.emit(self.values())
        self.updateDisplayPositions()

    def setStep(self, step):
        self._step = step

    @property
    def single_step(self):
        return self._step / self.scale

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            return

        size = self.rangeSliderSize()
        pos = self.getPos(event)
        if self.moving == "min":
            if pos <= self.handle_radius:
                self.display_min = self.handle_radius
            elif pos > self.display_max - self.handle_radius / 2:
                self.display_min = self.display_max - self.handle_radius / 2
            else:
                self.display_min = pos
        elif self.moving == "max":
            if pos >= size + self.handle_radius:
                self.display_max = size + self.handle_radius
            elif pos < self.display_min + self.handle_radius / 2:
                self.display_max = self.display_min + self.handle_radius / 2
            else:
                self.display_max = pos
        elif self.moving == "bar":
            width = self.start_display_max - self.start_display_min
            lower_part = self.start_pos - self.start_display_min
            upper_part = self.start_display_max - self.start_pos
            if pos + upper_part >= size + self.handle_radius:
                self.display_max = size + self.handle_radius
                self.display_min = self.display_max - width
            elif pos - lower_part <= self.handle_radius:
                self.display_min = self.handle_radius
                self.display_max = self.display_min + width
            else:
                self.display_min = pos - lower_part
                self.display_max = self.display_min + width

        self.updateValuesFromDisplay()

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return

        pos = self.getPos(event)
        top = self.rangeSliderSize() + self.handle_radius
        if event.button() == Qt.LeftButton:
            if not self.collapsed:
                if abs(self.display_min - pos) <= (self.handle_radius):
                    self.moving = "min"
                elif abs(self.display_max - pos) <= (self.handle_radius):
                    self.moving = "max"
                elif pos > self.display_min and pos < self.display_max:
                    self.moving = "bar"
                elif pos > self.display_max and pos < top:
                    self.display_max = pos
                    self.moving = "max"
                    self.updateValuesFromDisplay()
                elif pos < self.display_min and pos > self.handle_radius:
                    self.display_min = pos
                    self.moving = "min"
                    self.updateValuesFromDisplay()
            else:
                self.moving = "bar"
                if pos > self.handle_radius and pos < top:
                    self.display_max = pos
                    self.display_min = pos
        else:
            if self.collapsible:
                if self.collapsed:
                    self.expand()
                else:
                    self.collapse()
                self.collapsedChanged.emit(self.collapsed)

        self.start_display_min = self.display_min
        self.start_display_max = self.display_max
        self.start_pos = pos
        self.focused.emit()

    def mouseReleaseEvent(self, event):
        if self.isEnabled():
            if not (self.moving == "none"):
                self.valuesChanged.emit(self.values())
            self.moving = "none"

    def collapse(self):
        self.bc_min, self.bc_max = self.value_min, self.value_max
        midpoint = (self.value_max + self.value_min) / 2
        min_value = midpoint
        max_value = midpoint
        self.setSliderValues((min_value, max_value))
        self.collapsed = True

    def expand(self):
        _mid = (self.bc_max - self.bc_min) / 2
        min_value = self.value_min - _mid
        max_value = self.value_min + _mid
        if min_value < 0:
            min_value = 0
            max_value = self.bc_max - self.bc_min
        elif max_value > 1:
            max_value = 1
            min_value = max_value - (self.bc_max - self.bc_min)
        self.setSliderValues((min_value, max_value))
        self.collapsed = False

    def resizeEvent(self, event):
        self.updateDisplayPositions()
        self.resized.emit()

    def updateDisplayPositions(self):
        size = self.rangeSliderSize()
        range_min = int(size * self.value_min)
        range_max = int(size * self.value_max)
        self.display_min = range_min + self.handle_radius
        self.display_max = range_max + self.handle_radius
        self.update()

    def _data_to_slider_value(self, value):
        rmin, rmax = self.range()
        return (value - rmin) / self.scale

    def _slider_to_data_value(self, value):
        rmin, rmax = self.range()
        return rmin + value * self.scale

    @property
    def scale(self):
        return self.data_range_max - self.data_range_min

    def updateValuesFromDisplay(self):
        size = self.rangeSliderSize()
        val_min, val_max = self.sliderValues()
        if (self.moving == "min") or (self.moving == "bar"):
            scale_min = (self.display_min - self.handle_radius) / size
            ratio = round(scale_min / self.single_step)
            val_min = ratio * self.single_step
        if (self.moving == "max") or (self.moving == "bar"):
            scale_max = (self.display_max - self.handle_radius) / size
            ratio = round(scale_max / self.single_step)
            val_max = ratio * self.single_step
        self.setSliderValues((val_min, val_max))

    def getBarColor(self):
        return self.bar_color

    def setBarColor(self, barColor):
        self.bar_color = barColor

    barColor = Property(QColor, getBarColor, setBarColor)

    def getBackgroundColor(self):
        return self.background_color

    def setBackgroundColor(self, backgroundColor):
        self.background_color = backgroundColor

    backgroundColor = Property(QColor, getBackgroundColor, setBackgroundColor)

    @property
    def handle_width(self):
        return self.handle_radius * 2

    def getHandleColor(self):
        return self.handle_color

    def setHandleColor(self, handleColor):
        self.handle_color = handleColor

    handleColor = Property(QColor, getHandleColor, setHandleColor)

    def getHandleBorderColor(self):
        return self.handle_border_color

    def setHandleBorderColor(self, handleBorderColor):
        self.handle_border_color = handleBorderColor

    handleBorderColor = Property(
        QColor, getHandleBorderColor, setHandleBorderColor
    )

    def setEnabled(self, bool):
        super().setEnabled(bool)
        self.update()


class QHRangeSlider(QRangeSlider):
    """
    Horizontal Range Slider, extended from QRangeSlider

    Parameters
    ----------
    initial_values : 2-tuple, optional
        Initial min & max values of the slider, defaults to (0.2, 0.8)
    data_range : 2-tuple, optional
        Min and max of the slider range, defaults to (0, 1)
    step_size : float, optional
        Single step size for the slider, defaults to 1
    collapsible : bool
        Whether the slider is collapsible, defaults to True.
    collapsed : bool
        Whether the slider begins collapsed, defaults to False.
    parent : qtpy.QtWidgets.QWidget
        Parent widget.
    """

    def getPos(self, event):
        """Get event position.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.

        Returns
        -------
        position : int
            Relative horizontal position of the event.
        """
        return event.x()

    def paintEvent(self, event):
        """Paint the background, range bar and splitters.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.
        """
        painter, w, h = QPainter(self), self.width(), self.height()

        half_width = self.slider_width / 2
        halfdiff = h / 2 - half_width

        # Background
        painter.setPen(self.background_color)
        painter.setBrush(self.background_color)
        painter.drawRect(0, halfdiff, w, self.slider_width)

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        if self.collapsed:
            painter.drawRect(0, halfdiff, self.display_max, self.slider_width)
        else:
            painter.drawRect(
                self.display_min,
                halfdiff,
                self.display_max - self.display_min,
                self.slider_width,
            )

        painter.setRenderHints(QPainter.Antialiasing)
        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(
            self.display_min - self.handle_radius,
            h / 2 - self.handle_radius,
            self.handle_width,
            self.handle_width,
        )  # left
        painter.drawEllipse(
            self.display_max - self.handle_radius,
            h / 2 - self.handle_radius,
            self.handle_width,
            self.handle_width,
        )  # right

    def rangeSliderSize(self):
        """Width of the slider, in pixels

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical
            sliders).
        """
        return float(self.width() - self.handle_width)


class QVRangeSlider(QRangeSlider):
    """
    Vertical Range Slider, extended from QRangeSlider

    Parameters
    ----------
    initial_values : 2-tuple, optional
        Initial min & max values of the slider, defaults to (0.2, 0.8)
    data_range : 2-tuple, optional
        Min and max of the slider range, defaults to (0, 1)
    step_size : float, optional
        Single step size for the slider, defaults to 1
    collapsible : bool
        Whether the slider is collapsible, defaults to True.
    collapsed : bool
        Whether the slider begins collapsed, defaults to False.
    parent : qtpy.QtWidgets.QWidget
        Parent widget.
    """

    def getPos(self, event):
        """Get event position.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.

        Returns
        -------
        position : int
            Relative horizontal position of the event.
        """
        return self.height() - event.y()

    def paintEvent(self, event):
        """Paint the background, range bar and splitters.

        Parameters
        ----------
        event : qtpy.QEvent
            Event from the Qt context.
        """
        painter, w, h = QPainter(self), self.width(), self.height()

        half_width = self.slider_width / 2
        halfdiff = w / 2 - half_width
        # Background
        painter.setPen(self.background_color)
        painter.setBrush(self.background_color)
        painter.drawRect(halfdiff, 0, self.slider_width, h)

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        if self.collapsed:
            painter.drawRect(
                halfdiff,
                h - self.display_max,
                self.slider_width,
                self.display_max,
            )
        else:
            painter.drawRect(
                halfdiff,
                h - self.display_max,
                self.slider_width,
                self.display_max - self.display_min,
            )

        painter.setRenderHints(QPainter.Antialiasing)
        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(
            w / 2 - self.handle_radius,
            h - self.display_min - self.handle_radius,
            self.handle_width,
            self.handle_width,
        )  # upper
        painter.drawEllipse(
            w / 2 - self.handle_radius,
            h - self.display_max - self.handle_radius,
            self.handle_width,
            self.handle_width,
        )  # lower

    def rangeSliderSize(self):
        """Height of the slider, in pixels

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical
            sliders).
        """
        return float(self.height() - self.handle_width)
