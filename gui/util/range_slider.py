"""
Range slider, extended QWidget slider for napari-gui.
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class QRangeSlider(QtWidgets.QWidget):
    """
    QRangeSlider class, super class for QVRangeSlider and QHRangeSlider.
    """
    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, slider_range, values, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.bar_width = 16
        self.slider_width = 2
        self.emit_while_moving = 0
        self.moving = "none"
        self.old_scale_min = 0.0
        self.old_scale_max = 0.0
        self.scale = 0
        self.setMouseTracking(False)
        self.single_step = 0.0

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
        self.setEnabled(False)

        if slider_range:
            self.setRange(slider_range)
        else:
            self.setRange([0.0, 1.0, 0.01])
        if values:
            self.setValues(values)
        else:
            self.setValues([0.3, 0.6])

    def emitRange(self):
        if (self.old_scale_min != self.scale_min) or (self.old_scale_max != self.scale_max):
            self.rangeChanged.emit(self.scale_min, self.scale_max)
            self.old_scale_min = self.scale_min
            self.old_scale_max = self.scale_max
            # For debug purposes
            # if False:
            #     print("Range change:", self.scale_min, self.scale_max)

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
        values : 2-tuple of int
            Start and end of the range.
        """
        self.scale_min, self.scale_max = values
        #self.emitRange()
        self.updateDisplayValues()
        self.update()

    def mouseMoveEvent(self, event):
        if self.enabled:
            size = self.rangeSliderSize()
            pos = self.getPos(event)
            if self.moving == "min":
                if pos <= self.bar_width/2:
                    self.display_min = self.bar_width/2
                elif pos > self.display_max-self.bar_width/4:
                    self.display_min = self.display_max-self.bar_width/4
                else:
                    self.display_min = pos
            elif self.moving == "max":
                if pos >= size+self.bar_width/2:
                    self.display_max = size+self.bar_width/2
                elif pos < self.display_min+self.bar_width/4:
                   self.display_max = self.display_min+self.bar_width/4
                else:
                    self.display_max = pos
            elif self.moving == "bar":
                width = self.start_display_max - self.start_display_min
                part = self.start_pos - self.start_display_min
                if pos + (self.start_display_max - self.start_pos) >= size+self.bar_width/2:
                    self.display_max = size+self.bar_width/2
                    self.display_min = self.display_max - width
                elif pos - (self.start_pos - self.start_display_min) <= self.bar_width/2:
                    self.display_min = self.bar_width/2
                    self.display_max = self.display_min + width
                else:
                    self.display_min = pos - (self.start_pos - self.start_display_min)
                    self.display_max = self.display_min + width

            self.updateScaleValues()
            if self.emit_while_moving:
                self.emitRange()

    def mousePressEvent(self, event):
        if self.enabled:
            pos = self.getPos(event)
            if event.button() == QtCore.Qt.LeftButton:
                if not self.collapsed:
                    if abs(self.display_min - pos) <= (0.5 * self.bar_width):
                        self.moving = "min"
                    elif abs(self.display_max - pos) <= (0.5 * self.bar_width):
                        self.moving = "max"
                    elif (pos > self.display_min) and (pos < self.display_max):
                        self.moving = "bar"
                    elif (pos > self.display_max) and (pos < (self.rangeSliderSize() + 0.5 * self.bar_width)):
                        self.display_max = pos
                        self.moving = "max"
                        self.updateScaleValues()
                        if self.emit_while_moving:
                            self.emitRange()
                    elif (pos < self.display_min) and (pos > 0.5 * self.bar_width):
                        self.display_min = pos
                        self.moving = "min"
                        self.updateScaleValues()
                        if self.emit_while_moving:
                            self.emitRange()
                    else:
                        self.moving = "bar"
            else:
                if self.collapsed:
                    # print("collapsed already")
                    self.expand()
                    self.collapsed = False
                else:
                    # print("not collapsed")
                    self.collapse()
                    self.collapsed = True

            self.start_display_min = self.display_min
            self.start_display_max = self.display_max
            self.start_pos = pos

    def collapse(self):
        self.bc_min, self.bc_max = self.scale_min, self.scale_max
        while int(self.scale_min) < int(self.scale_max):
            step = (self.scale_max - self.scale_min) / 2
            self.setValues([self.scale_min+step, self.scale_max-step])

    def expand(self):
        self.setValues([self.bc_min, self.bc_max])

    def mouseReleaseEvent(self, event):
        if self.enabled:
            if not (self.moving == "none"):
                self.emitRange()
            self.moving = "none"

    def resizeEvent(self, event):
        self.updateDisplayValues()

    def setRange(self, slider_range):
        self.start = slider_range[0]
        self.scale = slider_range[1] - slider_range[0]
        self.single_step = slider_range[2]

    def setEmitWhileMoving(self, flag):
        if flag:
            self.emit_while_moving = 1
        else:
            self.emit_while_moving = 0

    def updateDisplayValues(self):
        size = self.rangeSliderSize()
        self.display_min = int(size * (self.scale_min - self.start) / self.scale) + self.bar_width/2
        self.display_max = int(size * (self.scale_max - self.start) / self.scale) + self.bar_width/2

    def updateScaleValues(self):
        size = self.rangeSliderSize()
        if (self.moving == "min") or (self.moving == "bar"):
            self.scale_min = self.start + (self.display_min - self.bar_width/2) / float(size) * self.scale
            self.scale_min = float(round(self.scale_min / self.single_step)) * self.single_step
        if (self.moving == "max") or (self.moving == "bar"):
            self.scale_max = self.start + (self.display_max - self.bar_width/2) / float(size) * self.scale
            self.scale_max = float(round(self.scale_max / self.single_step)) * self.single_step
        self.updateDisplayValues()
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
    parent : PyQt5.QtWidgets.QWidget
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
        event : PyQt5.QtCore.QEvent
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
        event : PyQt5.QtCore.QEvent
            Event from the Qt context.
        """
        painter, w, h = QtGui.QPainter(self), self.width(), self.height()

        # Background
        painter.setPen(QtCore.Qt.gray)
        painter.setBrush(QtCore.Qt.transparent)
        painter.drawRect(2, 2, w - 4, h - 4)

        # Range Bar
        painter.setPen(QtCore.Qt.darkGray)
        painter.setBrush(QtCore.Qt.darkCyan)
        painter.drawRect(self.display_min - 1, 5, self.display_max - self.display_min + 2, h - 10)

        # Splitters
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.gray)
        painter.drawRect(self.display_min - self.bar_width, 1, self.bar_width, h - 2)  # left
        painter.drawRect(self.display_max, 1, self.bar_width, h - 2)  # right

    def rangeSliderSize(self):
        """Size of the slider.

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical sliders).
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
    parent : PyQt5.QtWidgets.QWidget
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
        event : PyQt5.QtCore.QEvent
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
        event : PyQt5.QtCore.QEvent
            Event from the Qt context.
        """
        painter, w, h = QtGui.QPainter(self), self.width(), self.height()

        # Background
        painter.setPen(QtCore.Qt.lightGray)
        painter.setBrush(QtCore.Qt.lightGray)
        painter.drawRect(w/2-self.slider_width/2, self.bar_width/2, self.slider_width, h-self.bar_width)

        # Range Bar
        painter.setPen(self.bar_color)
        painter.setBrush(self.bar_color)
        painter.drawRect(w/2-self.slider_width/2, h-self.display_max, self.slider_width, self.display_max-self.display_min)

        # Splitters
        painter.setPen(self.handle_border_color)
        painter.setBrush(self.handle_color)
        painter.drawEllipse(w/2-self.bar_width/2, h-self.display_min-self.bar_width/2, self.bar_width, self.bar_width) # upper
        painter.drawEllipse(w/2-self.bar_width/2, h-self.display_max-self.bar_width/2, self.bar_width, self.bar_width) # lower

    def rangeSliderSize(self):
        """Size of the slider.

        Returns
        -------
        size : int
            Slider bar length (horizontal sliders) or height (vertical sliders).
        """
        return float(self.height() - self.bar_width)

    def setEnabled(self, bool):
        if bool:
            self.enabled = True
            self.bar_color = QtGui.QColor(0, 153, 255)
            self.handle_color = QtCore.Qt.white
            self.handle_border_color = QtCore.Qt.lightGray
        else:
            self.enabled = False
            self.bar_color = QtCore.Qt.gray
            self.handle_color = QtCore.Qt.gray
            self.handle_border_color = QtCore.Qt.gray
        self.update()
