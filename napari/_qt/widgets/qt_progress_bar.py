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


def get_pbar(nest_under=None, **kwargs):
    """Adds ProgressBar to viewer Activity Dock and returns it.
    If nest_under is valid ProgressBar, nests new bar underneath
    parent in a ProgressBarGroup

    Parameters
    ----------
    nest_under : Optional[ProgressBar]
        parent ProgressBar to nest under, by default None

    Returns
    -------
    ProgressBar
        progress bar to associate with iterable
    """
    from ..qt_main_window import _QtMainWindow

    current_window = _QtMainWindow.current()
    if current_window is None:
        return
    viewer_instance = current_window.qt_viewer
    pbar = ProgressBar(**kwargs)
    pbr_layout = viewer_instance.activityDock.widget().layout()

    if nest_under is None:
        pbr_layout.addWidget(pbar)
    else:
        # this is going to be nested, remove separators
        # as the group will have its own
        parent_pbar = nest_under._pbar
        current_pbars = [parent_pbar, pbar]
        remove_separators(current_pbars)

        parent_widg = parent_pbar.parent()
        if isinstance(parent_widg, ProgressBarGroup):
            nested_layout = parent_widg.layout()
        else:
            new_group = ProgressBarGroup(nest_under._pbar)
            nested_layout = new_group.layout()
            pbr_layout.addWidget(new_group)
        new_pbar_index = nested_layout.count() - 1
        nested_layout.insertWidget(new_pbar_index, pbar)

    return pbar


def remove_separators(current_pbars):
    """Remove any existing line separators from current_pbars
    as they will get a separator from the group

    Parameters
    ----------
    current_pbars : List[ProgressBar]
        parent and new progress bar to remove separators from
    """
    for current_pbar in current_pbars:
        line_widg = current_pbar.findChild(QFrame, "QtCustomTitleBarLine")
        if line_widg:
            current_pbar.layout().removeWidget(line_widg)
            line_widg.hide()
            line_widg.deleteLater()
