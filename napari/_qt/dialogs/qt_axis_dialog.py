from typing import Optional

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ...utils.translations import trans


class QAxisDialog(QDialog):
    """Blocking dialog for axis selection, it assumes that axis are from 0 to ndim - 1.

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
        self.setObjectName('QAxisDialog')
        self._ndim = ndim

        self._combobox = QComboBox(self)
        self._combobox.addItems([str(v) for v in range(ndim)])
        self._combobox.setCurrentIndex(0)

        self._confirm_btn = QPushButton()
        self._confirm_btn.setText(trans._("Confirm"))
        self._confirm_btn.clicked.connect(self.accept)

        self._cancel_btn = QPushButton()
        self._cancel_btn.setText(trans._("Cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        self._cancel_btn.clicked.connect(self.close)

        layout = QGridLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel(trans._("Axis:")), 0, 0)
        layout.addWidget(self._combobox, 0, 1)
        layout.addWidget(self._confirm_btn, 1, 0)
        layout.addWidget(self._cancel_btn, 1, 1)

    @property
    def value(self) -> int:
        axis = self._combobox.currentIndex()
        return axis

    @value.setter
    def value(self, axis: int) -> int:
        return self._combobox.setCurrentIndex(axis)
