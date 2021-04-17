import inspect
from typing import Iterable, Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)
from tqdm import tqdm

from .._qt.utils import get_viewer_instance


def get_pbar(viewer_instance):
    pbar = ProgressBar()
    viewer_instance.activityDock.widget().layout.addWidget(pbar.baseWidget)

    return pbar


def get_calling_function_name(max_depth: int):
    """Inspect stack up to max_depth and return first function name outside of progress.py"""
    for finfo in inspect.stack()[2:max_depth]:
        if not finfo.filename.endswith("progress.py"):
            return finfo.function

    return None


class progress(tqdm):
    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:

        # check if there's a napari viewer instance
        viewer = get_viewer_instance()
        if viewer is not None:
            kwargs['gui'] = True
        super().__init__(iterable, desc, total, *args, **kwargs)
        if viewer is None:
            return
        self.viewer = viewer

        self._pbar = get_pbar(viewer)
        if self.total is not None:
            self._pbar.setRange(self.n, self.total)
            self._pbar._set_value(self.n)
        else:
            self._pbar.setRange(0, 0)

        if desc:
            self._pbar._set_description(desc)
        else:
            desc = get_calling_function_name(max_depth=5)
            if desc:
                self._pbar._set_description(desc)

        self.show()
        QApplication.processEvents()

    def display(self, msg: str = None, pos: int = None) -> None:
        """Update the display."""
        if not self.viewer:
            return super().display(msg=msg, pos=pos)

        self._pbar._set_value(self.n)
        QApplication.processEvents()

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)

        self._pbar._set_description(self.desc)

    def hide(self):
        """Hide the progress bar"""
        self._pbar.baseWidget.hide()

    def show(self):
        """Show the progress bar"""
        self._pbar.baseWidget.show()

    def close(self):
        """Closes and deletes the progress bar widget"""
        super().close()
        self._pbar.baseWidget.close()


class ProgressBar:
    def __init__(self) -> None:
        self.baseWidget = QWidget()
        self.baseWidget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pbar = QProgressBar()
        self.label = QLabel()

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pbar)
        self.baseWidget.setLayout(layout)

    def setRange(self, min, max):
        self.pbar.setRange(min, max)

    def _set_value(self, value):
        self.pbar.setValue(value)

    def _get_value(self):
        return self.pbar.value()

    def _set_description(self, desc):
        self.label.setText(desc)
