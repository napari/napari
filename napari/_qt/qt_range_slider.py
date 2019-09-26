"""
Range slider, extended QWidget slider for napari.
"""

from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QWidget


class QRangeSlider(QWidget):
    """
    QRangeSlider class, super class for QVRangeSlider and QHRangeSlider.
    """

    rangeChanged = QtCore.Signal(float, float)
    collapsedChanged = QtCore.Signal(bool)
    focused = QtCore.Signal()

    def __init__(self, slider_range, values, parent=None):
        QWidget.__init__(self, parent)
        self.bar_width = 16
        self.slider_width = 8
        self.emit_while_moving = 0
        self.moving = "none"
        self.old_scale_min = 0.0
        self.old_scale_max = 0.0
        self.scale = 0
        self.setMouseTracking(False)
        self.single_step = 0.0

        self.default_collapse_logic = True
        self.collapsable = True
        self.collapsed = False
        self.prev_moving = None
        self.bc_min = None
        self.bc_max = None

        # Variables initialized in methods
        self.scale_min = None
        self.scale_max = None
        self.start_display_min = None
        self.start_display_max = None
        self.start_pos = None
        self.display_min = None
        self.display_max = None

        self.setBarColor(QtGui.QColor(200, 200, 200))
        self.setBackgroundColor(QtGui.QColor(100, 100, 100))
        self.setHandleColor(QtGui.QColor(200, 200, 200))
        self.setHandleBorderColor(QtGui.QColor(200, 200, 200))

        self.setEnabled(True)

        if slider_range:
            self.setRange(slider_range)
        else:
            self.setRange((0.0, 1.0, 0.01))
        if values:
            self.setValues(values)
        else:
            self.setValues((0.3, 0.7))

    def emitRange(self):
        change_min = self.old_scale_min != self.scale_min
        change_max = self.old_scale_max != self.scale_max
        if change_min or change_max:
            self.rangeChanged.emit(self.scale_min, self.scale_max)
            self.old_scale_min = self.scale_min
            self.old_scale_max = self.scale_max
            # For debug purposes
            # if False:
            #     print("Range change:", self.scale_min, self.scale_max)

    def emitCollapse(self, collapsed_state):
        self.collapsedChanged.emit(collapsed_state)

    def getValues(self):
        """Values of the range bar.

        Returns
        -------
        values : 2-tuple of int
            Start and end of the range.
        """
        return [self.scale_min, self.scale_max]

    def setValues(self, values):
        """Set values of the range bar.

        Parameters
        ----------
        values : 2-tuple of float or int
            Start and end of the range.
        """
        if values is not None:
            self.scale_min, self.scale_max = values
            if self.scale_min is None:
                self.scale_min = self.start
            if self.scale_max is None:
                self.scale_max = self.end
        else:
            self.scale_min = self.start
            self.scale_max = self.end
        self.emitRange()
        self.updateDisplayValues()
        self.update()

    def setValue(self, value):
        """Set values of the range bar.

        Parameters
        ----------
        value : float | int
            Value to used when collapsed
        """
        self.setValues((value, value))

    def mouseMoveEvent(self, event):
        if self.enabled:
            size = self.rangeSliderSize()
            pos = self.getPos(event)
            if self.moving == "min":
                if pos <= self.bar_width / 2:
                    self.display_min = self.bar_width / 2
                elif pos > self.display_max - self.bar_width / 4:
                    self.display_min = self.display_max - self.bar_width / 4
                else:
                    self.display_min = pos
            elif self.moving == "max":
                if pos >= size + self.bar_width / 2:
                    self.display_max = size + self.bar_width / 2
                elif pos < self.display_min + self.bar_width / 4:
                    self.display_max = self.display_min + self.bar_width / 4
                else:
                    self.display_max = pos
            elif self.moving == "bar":
                width = self.start_display_max - self.start_display_min
                lower_part = self.start_pos - self.start_display_min
                upper_part = self.start_display_max - self.start_pos
                if pos + upper_part >= size + self.bar_width / 2:
                    self.display_max = size + self.bar_width / 2
                    self.display_min = self.display_max - width
                elif pos - lower_part <= self.bar_width / 2:
                    self.display_min = self.bar_width / 2
                    self.display_max = self.display_min + width
                else:
                    self.display_min = pos - lower_part
                    self.display_max = self.display_min + width

            self.updateScaleValues()
            if self.emit_while_moving:
                self.emitRange()

    def mousePressEvent(self, event):
        if self.enabled:
            pos = self.getPos(event)
            top = self.rangeSliderSize() + self.bar_width / 2
            if event.button() == QtCore.Qt.LeftButton:
                if not self.collapsed:
                    if abs(self.display_min - pos) <= (self.bar_width / 2):
                        self.moving = "min"
                    elif abs(self.display_max - pos) <= (self.bar_width / 2):
                        self.moving = "max"
                    elif pos > self.display_min and pos < self.display_max:
                        self.moving = "bar"
                    elif pos > self.display_max and pos < top:
                        self.display_max = pos
                        self.moving = "max"
                        self.updateScaleValues()
                        if self.emit_while_moving:
                            self.emitRange()
                    elif pos < self.display_min and pos > self.bar_width / 2:
                        self.display_min = pos
                        self.moving = "min"
                        self.updateScaleValues()
                        if self.emit_while_moving:
                            self.emitRange()
                else:
                    self.moving = "bar"
                    if pos > self.bar_width / 2 and pos < top:
                        self.display_max = pos
                        self.display_min = pos
                        self.updateScaleValues()
                        if self.emit_while_moving:
                            self.emitRange()
            else:
                if self.collapsable:
                    if self.collapsed:
                        # print("collapsed already")
                        self.expand()
                    else:
                        # print("not collapsed")
                        self.collapse()
                    self.emitCollapse(self.collapsed)

            self.start_display_min = self.display_min
            self.start_display_max = self.display_max
            self.start_pos = pos
        self.focused.emit()

    def collapse(self):
        if self.default_collapse_logic:
            self.bc_min, self.bc_max = self.scale_min, self.scale_max
            min_value = (self.scale_max + self.scale_min) / 2
            max_value = (self.scale_max + self.scale_min) / 2
            self.setValues((min_value, max_value))
        else:
            # self.setValues((self.scale_min, self.scale_max))
            self.update()
        self.collapsed = True

    def expand(self):
        if self.default_collapse_logic:
            min_value = self.scale_min - (self.bc_max - self.bc_min) / 2
            max_value = self.scale_min + (self.bc_max - self.bc_min) / 2
            if min_value < self.start:
                min_value = self.start
                max_value = min_value + self.bc_max - self.bc_min
            elif max_value > self.end:
                max_value = self.end
                min_value = max_value - (self.bc_max - self.bc_min)
            self.setValues((min_value, max_value))
        else:
            # self.setValues((self.scale_min, self.scale_max))
            self.update()
        self.collapsed = False

    def mouseReleaseEvent(self, event):
        if self.enabled:
            if not (self.moving == "none"):
                self.emitRange()
            self.moving = "none"

    def resizeEvent(self, event):
        self.updateDisplayValues()

    def setRange(self, slider_range):
        self.start, self.end, self.single_step = slider_range
        self.scale = self.end - self.start

    def setEmitWhileMoving(self, flag):
        if flag:
            self.emit_while_moving = 1
        else:
            self.emit_while_moving = 0

    def updateDisplayValues(self):
        size = self.rangeSliderSize()
        if self.scale == 0:
            range_min = 0
            range_max = 0
        else:
            range_min = int(size * (self.scale_min - self.start) / self.scale)
            range_max = int(size * (self.scale_max - self.start) / self.scale)
        self.display_min = range_min + self.bar_width / 2
        self.display_max = range_max + self.bar_width / 2

    def updateScaleValues(self):
        size = self.rangeSliderSize()
        if (self.moving == "min") or (self.moving == "bar"):
            ratio = (self.display_min - self.bar_width / 2) / float(size)
            scale_min = self.start + ratio * self.scale
            ratio = float(round(scale_min / self.single_step))
            self.scale_min = ratio * self.single_step
        if (self.moving == "max") or (self.moving == "bar"):
            ratio = (self.display_max - self.bar_width / 2) / float(size)
            scale_max = self.start + ratio * self.scale
            ratio = float(round(scale_max / self.single_step))
            self.scale_max = ratio * self.single_step
        self.updateDisplayValues()
        self.update()

    def getBarColor(self):
        return self.bar_color

    def setBarColor(self, barColor):
        self.bar_color = barColor

    barColor = QtCore.Property(QtGui.QColor, getBarColor, setBarColor)

    def getBackgroundColor(self):
        return self.background_color

    def setBackgroundColor(self, backgroundColor):
        self.background_color = backgroundColor

    backgroundColor = QtCore.Property(
        QtGui.QColor, getBackgroundColor, setBackgroundColor
    )

    def getHandleColor(self):
        return self.handle_color

    def setHandleColor(self, handleColor):
        self.handle_color = handleColor

    handleColor = QtCore.Property(QtGui.QColor, getHandleColor, setHandleColor)

    def getHandleBorderColor(self):
        return self.handle_border_color

    def setHandleBorderColor(self, handleBorderColor):
        self.handle_border_color = handleBorderColor

    handleBorderColor = QtCore.Property(
        QtGui.QColor, getHandleBorderColor, setHandleBorderColor
    )

    def setEnabled(self, bool):
        if bool:
            self.enabled = True
        else:
            self.enabled = False
        self.update()


class QHRangeSlider(QRangeSlider):
    """
    Horizontal Range Slider, extended from QRangeSlider

    Parameters
    ----------
    slider_range : 3-tuple of int
        Min, max, and step of the slider.
    values : 2-tuple of int
        Start and end of the slider range.
    parent : qtpy.QtWidgets.QWidget
        Parent widget.
    """

    def __init__(self, slider_range=None, values=None, parent=None):
        QRangeSlider.__init__(self, slider_range, values, parent)
        if not parent:
            self.setGeometry(200, 200, 200, 20)

    def getPos(self, event):
        """Get event position.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
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
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter, w, h = QtGui.QPainter(self), self.width(), self.height()

        # Background
        painter.setPen(self.background_color)
        painter.setBrush(self.background_color)
        painter.drawRect(
            0, h / 2 - self.slider_width / 2, w, self.slider_width
        )

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        if self.collapsed:
            painter.drawRect(
                0,
                h / 2 - self.slider_width / 2,
                self.display_max,
                self.slider_width,
            )
        else:
            painter.drawRect(
                self.display_min,
                h / 2 - self.slider_width / 2,
                self.display_max - self.display_min,
                self.slider_width,
            )

        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(
            self.display_min - self.bar_width / 2,
            h / 2 - self.bar_width / 2,
            self.bar_width,
            self.bar_width,
        )  # left
        painter.drawEllipse(
            self.display_max - self.bar_width / 2,
            h / 2 - self.bar_width / 2,
            self.bar_width,
            self.bar_width,
        )  # right

    def rangeSliderSize(self):
        """Size of the slider.

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical
            sliders).
        """
        return float(self.width() - self.bar_width)


class QVRangeSlider(QRangeSlider):
    """
    Vertical Range Slider, extended from QRangeSlider

    Parameters
    ----------
    slider_range : 3-tuple of int
        Min, max, and step of the slider.
    values : 2-tuple of int
        Start and end of the slider range.
    parent : qtpy.QtWidgets.QWidget
        Parent widget.
    """

    def __init__(self, slider_range=None, values=None, parent=None):
        QRangeSlider.__init__(self, slider_range, values, parent)
        if not parent:
            self.setGeometry(200, 200, 20, 200)

    def getPos(self, event):
        """Get event position.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
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
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter, w, h = QtGui.QPainter(self), self.width(), self.height()

        # Background
        painter.setPen(self.background_color)
        painter.setBrush(self.background_color)
        painter.drawRect(
            w / 2 - self.slider_width / 2, 0, self.slider_width, h
        )

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        if self.collapsed:
            painter.drawRect(
                w / 2 - self.slider_width / 2,
                h - self.display_max,
                self.slider_width,
                self.display_max,
            )
        else:
            painter.drawRect(
                w / 2 - self.slider_width / 2,
                h - self.display_max,
                self.slider_width,
                self.display_max - self.display_min,
            )

        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(
            w / 2 - self.bar_width / 2,
            h - self.display_min - self.bar_width / 2,
            self.bar_width,
            self.bar_width,
        )  # upper
        painter.drawEllipse(
            w / 2 - self.bar_width / 2,
            h - self.display_max - self.bar_width / 2,
            self.bar_width,
            self.bar_width,
        )  # lower

    def rangeSliderSize(self):
        """Size of the slider.

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical
            sliders).
        """
        return float(self.height() - self.bar_width)
