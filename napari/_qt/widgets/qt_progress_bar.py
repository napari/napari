from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


def get_pbar(current_group_ref, **kwargs):
    """Adds ProgressBar to viewer Activity Dock and returns it.

    Parameters
    ----------
    viewer_instance : qtViewer
        current napari qtViewer instance

    Returns
    -------
    ProgressBar
        progress bar to associate with current iterable
    """
    from ..qt_main_window import _QtMainWindow

    current_window = _QtMainWindow.current()
    if current_window is None:
        return
    viewer_instance = current_window.qt_viewer
    pbar = ProgressBar(**kwargs)
    pbr_layout = viewer_instance.activityDock.widget().layout()

    if current_group_ref:
        group_widg = current_group_ref()
        group_layout = group_widg.layout()
        group_layout.addWidget(pbar)
    else:
        pbr_group = ProgressBarGroup(pbar)
        pbr_layout.addWidget(pbr_group)

    return pbar


class ProgressBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pbar = QProgressBar()
        self.description_label = QLabel()
        self.eta_label = QLabel()

        layout = QHBoxLayout()
        layout.addWidget(self.description_label)
        layout.addWidget(self.pbar)
        layout.addWidget(self.eta_label)
        self.setLayout(layout)

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
    def __init__(self, pbar, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        pbr_group_layout = QVBoxLayout()
        pbr_group_layout.addWidget(pbar)
        self.setLayout(pbr_group_layout)
