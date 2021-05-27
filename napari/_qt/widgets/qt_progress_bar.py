from qtpy import QtCore
from qtpy.QtWidgets import (  # QVBoxLayout,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)


def get_pbar(nested_ref, **kwargs):
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

    if nested_ref:
        parent_ref, count = nested_ref
        parent_pbar_widg = parent_ref()
        index_of_parent = pbr_layout.indexOf(parent_pbar_widg)
        new_index = index_of_parent + count
        pbr_layout.insertWidget(new_index, pbar)
    else:
        pbr_layout.addWidget(pbar)

        line = QFrame(parent=pbar)
        line.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        line.setObjectName("QtCustomTitleBarLine")
        line.setFixedHeight(1)
        pbr_layout.addWidget(line)
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


# class ProgressBarGroup(QWidget):
#     def __init__(self, pbar, parent=None) -> None:
#         super().__init__(parent)
#         self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

#         pbr_group_layout = QVBoxLayout()
#         pbr_group_layout.addWidget(pbar)
#         pbr_group_layout.setContentsMargins(0, 0, 0, 0)
#         self.setLayout(pbr_group_layout)
