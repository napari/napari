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


class QtLabeledProgressBar(QWidget):
    """QProgressBar with QLabels for description and ETA."""

    def __init__(self, parent=None, prog=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.prog = prog

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

    def setValue(self, value):
        self.pbar.setValue(value)
        QApplication.processEvents()

    def setDescription(self, value):
        self.description_label.setText(value)
        QApplication.processEvents()

    def _set_value(self, event):
        self.setValue(event.value)

    def _get_value(self):
        return self.pbar.value()

    def _set_description(self, event):
        self.setDescription(event.value)

    def _make_indeterminate(self, event):
        self.setRange(0, 0)

    def _set_eta(self, event):
        self.eta_label.setText(event.value)

    def _close(self, event):
        super().close()


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
