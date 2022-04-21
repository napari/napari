from typing import Optional

from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from ...utils.translations import trans
from ..dialogs.qt_modal import QtPopup


class QAxisPopup(QtPopup):
    """Blocking popup for axis selection, it assumes that axis are from 0 to ndim - 1.

    Parameters
    ----------
    ndim : int
        number of dimensions, popup axis options are from 0 to ndim - 1
    parent : QWidget, optional
        parent widget, by default None

    Attributes
    ----------
    value : Optional[int]
        selected axis or None if no axis was selected
    """

    def __init__(self, ndim: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName('QAxisPopup')
        self._ndim = ndim

        self._combobox = QComboBox(self)
        self._combobox.addItems([str(v) for v in range(ndim)])
        self._combobox.setCurrentIndex(-1)
        self._combobox.currentIndexChanged.connect(self._on_value_changed)

        layout = QHBoxLayout()
        self.frame.setLayout(layout)

        layout.addWidget(QLabel(trans._("Axis:")))
        layout.addWidget(self._combobox)

    @property
    def value(self) -> Optional[int]:
        axis = self._combobox.currentIndex()
        if axis < 0:
            # no item selected
            return None
        return axis

    @value.setter
    def value(self, axis: int) -> int:
        return self._combobox.setCurrentIndex(axis)

    def _on_value_changed(self) -> None:
        self.close()
