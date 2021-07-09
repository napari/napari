from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class ProgressBar(QWidget):
    """QProgressBar with QLabels for description and ETA."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pbar = QProgressBar()
        self.description_label = QLabel()
        self.eta_label = QLabel()
        base_layout = QVBoxLayout()

        pbar_layout = QHBoxLayout()
        pbar_layout.addWidget(self.description_label)
        pbar_layout.addWidget(self.pbar)
        pbar_layout.addWidget(self.eta_label)
        base_layout.addLayout(pbar_layout)

        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        line.setFixedHeight(1)
        base_layout.addWidget(line)

        self.setLayout(base_layout)

    def setRange(self, min, max):
        self.pbar.setRange(min, max)

    def _set_value(self, value):
        self.pbar.setValue(value)
        QApplication.processEvents()

    def _get_value(self):
        return self.pbar.value()

    def _set_description(self, desc):
        self.description_label.setText(desc)
        QApplication.processEvents()

    def _set_eta(self, eta):
        self.eta_label.setText(eta)


class ProgressBarGroup(QWidget):
    """One or more QProgressBars with a QFrame line separator at the bottom"""

    def __init__(self, pbar, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        pbr_group_layout = QVBoxLayout()
        pbr_group_layout.addWidget(pbar)
        pbr_group_layout.setContentsMargins(0, 0, 0, 0)

        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        line.setFixedHeight(1)
        pbr_group_layout.addWidget(line)

        self.setLayout(pbr_group_layout)
