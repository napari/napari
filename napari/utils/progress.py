from typing import Iterable, Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from .._qt.utils import get_viewer_instance


def get_pbar():
    pbar = ProgressBar()
    viewer_instance = get_viewer_instance()
    viewer_instance.activityDock.widget().layout.addWidget(pbar.baseWidget)

    return pbar


class progress:
    def __init__(
        self, iterable: Optional[Iterable] = None, total: Optional[int] = None
    ) -> None:
        self._iterable = iterable
        self._pbar = get_pbar()

        if iterable is not None:  # iterator takes priority over total
            try:
                self._total = len(iterable) - 1
            except TypeError:  # generator (total needed)
                self._total = total if total is not None else 0
        else:
            if total is not None:
                self._iterable = range(total + 1)
                self._total = total
            else:
                self._total = 0

        self._pbar._set_total(self._total)
        # TODO: Find calling function and set to that
        self._pbar._set_description("Test Description")

        QApplication.processEvents()

    def __iter__(self):
        for n, i in enumerate(self._iterable):
            self._pbar._set_value(n)
            yield i

    def update(self, val, desc=None):
        """Update progress bar with new value and, optionally, a description.

        Parameters
        ----------
        val : int
            new value for progress bar
        desc : str, optional
            description to display on progress bar, by default None
        """
        if val > self._total:
            # exceeded total, become indeterminate
            self._pbar._set_total(0)
        else:
            self._pbar._set_value(val)

        if desc:
            self._pbar._set_description(desc)
        QApplication.processEvents()

    def hide(self):
        """Hide the progress bar"""
        self._pbar.baseWidget.hide()

    def show(self):
        """Show the progress bar"""
        self._pbar.baseWidget.show()

    def delete(self):
        """Delete the progress bar widget"""
        self._pbar.baseWidget.close()


class ProgressBar:
    def __init__(self, description="") -> None:
        self.baseWidget = QWidget()
        self.baseWidget.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pbar = QProgressBar()
        self.label = QLabel(description)

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pbar)
        self.baseWidget.setLayout(layout)

    def _set_total(self, total):
        if total > 0:
            self.pbar.setMaximum(total)
        else:
            self.pbar.setRange(0, 0)

    def _set_value(self, value):
        self.pbar.setValue(value)

    def _set_description(self, desc):
        self.label.setText(desc)
