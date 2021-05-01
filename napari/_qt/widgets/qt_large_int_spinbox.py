import logging

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDoubleSpinBox

logger = logging.getLogger(__name__)


class QtLargeIntSpinBox(QDoubleSpinBox):
    """
    A class for integer spinboxes backed by a double (i.e. float64).

    Qt's built-in ``QSpinBox`` is backed by a signed 32-bit integer.
    This could become limiting, particularly in large dense segmentations.
    This class behaves like a ``QSpinBox`` backed by a signed 54-bit integer.

    Instances are associated with a ``numpy.dtype``
    which can further constrain their maximum range,
    and ensures that values are clamped and cast correctly
    when using setter and getter methods.
    """

    MAX_FLOAT64_INT = 2 ** 53
    MIN_FLOAT64_INT = -MAX_FLOAT64_INT

    valueChanged = Signal(int)

    def __init__(self, dtype=int, parent=None):
        super().__init__(parent)
        self._dtype = np.dtype(dtype)
        if not np.issubdtype(self._dtype, np.integer):
            raise ValueError(f"Spinbox dtype must be integral, got {dtype}")
        iinfo = np.iinfo(self._dtype)

        self.setSingleStep(1)
        super().setDecimals(0)

        self._min = self._dtype.type(max(iinfo.min, self.MIN_FLOAT64_INT))
        self._max = self._dtype.type(min(iinfo.max, self.MAX_FLOAT64_INT))
        self.setRange(self._min, self._max)

    def _cast(self, value):
        actual_val = max(self._min, min(self._max, self._dtype.type(value)))
        if actual_val != value:
            logger.warning(
                "Value %s is not representable by %s(%s); using %s",
                value,
                type(self).__qualname__,
                self._dtype,
                actual_val,
            )
        return actual_val

    def setSingleStep(self, val):
        return super().setSingleStep(int(val))

    def singleStep(self):
        return int(super().singleStep())

    def stepBy(self, steps: int) -> None:
        super().stepBy(steps)
        self.valueChanged.emit(self.value())

    def setDecimals(self):
        raise NotImplementedError

    def setValue(self, value):
        val = self._cast(value)
        super().setValue(val)
        self.valueChanged.emit(val)

    def value(self):
        return self._cast(super().value())

    def setMinimum(self, value):
        super().setMinimum(self._cast(value))

    def minimum(self):
        return self._cast(super().minimum())

    def setMaximum(self, value):
        super().setMaximum(self._cast(value))

    def maximum(self):
        return self._cast(super().maximum())

    def setRange(self, min_, max_):
        self.setMinimum(min_)
        self.setMaximum(max_)
