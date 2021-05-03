from qtpy.QtCore import QSize, Signal
from qtpy.QtGui import QFontMetrics, QValidator
from qtpy.QtWidgets import QAbstractSpinBox, QStyle, QStyleOptionSpinBox


class AnyIntValidator(QValidator):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def validate(self, input: str, pos: int):
        if input.isnumeric():
            return QValidator.Acceptable, input, len(input)
        else:
            return QValidator.Invalid, input, len(input)


class QtLargeIntSpinBox(QAbstractSpinBox):
    """An integer spinboxes backed by unbound python integer

    Qt's built-in ``QSpinBox`` is backed by a signed 32-bit integer.
    This could become limiting, particularly in large dense segmentations.
    This class behaves like a ``QSpinBox`` backed by an unbound python int.

    Does not yet support "prefix", "suffix" or "specialValue" like QSpinBox.
    """

    textChanged = Signal(str)
    valueChanged = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value: int = 0
        self._minimum: int = 0
        self._maximum: int = 2 ** 64 - 1
        self._single_step: int = 1
        validator = AnyIntValidator(self)
        self.lineEdit().setValidator(validator)
        self.lineEdit().textChanged.connect(self._editor_text_changed)
        self.setValue(0)

    def value(self):
        return self._value

    def setValue(self, value):
        self._value, old = self._bound(int(value)), self._value
        self._updateEdit()
        self.update()
        if self._value != old:
            self.textChanged.emit(self.lineEdit().displayText())
            self.valueChanged.emit(self._value)

    def _updateEdit(self):
        new_text = str(self._value)
        if self.lineEdit().text() == new_text:
            return
        self.lineEdit().setText(new_text)

    def singleStep(self):
        return self._single_step

    def setSingleStep(self, step):
        self._single_step = int(step)

    def minimum(self):
        return self._minimum

    def setMinimum(self, min):
        self._minimum = int(min)

    def maximum(self):
        return self._maximum

    def setMaximum(self, max):
        self._maximum = int(max)

    def setRange(self, minimum, maximum):
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def stepBy(self, steps: int) -> None:
        step = self._single_step
        self.setValue(self._bound(self._value + (step * steps)))

    def _editor_text_changed(self, t):
        state, *_ = self.validate(t, self.lineEdit().cursorPosition())
        if state == QValidator.Acceptable:
            self.setValue(int(t))

    def _bound(self, value):
        return max(self._minimum, min(self._maximum, value))

    def stepEnabled(self):
        flags = QAbstractSpinBox.StepNone
        if self.isReadOnly():
            return flags
        if self._value < self._maximum:
            flags |= QAbstractSpinBox.StepUpEnabled
        if self._value > self._minimum:
            flags |= QAbstractSpinBox.StepDownEnabled
        return flags

    def sizeHint(self):
        self.ensurePolished()
        fm = QFontMetrics(self.font())
        h = self.lineEdit().sizeHint().height()
        w = fm.horizontalAdvance(str(self._value)) + 3
        w = max(36, w)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)
        hint = QSize(w, h)
        return self.style().sizeFromContents(
            QStyle.CT_SpinBox, opt, hint, self
        )
