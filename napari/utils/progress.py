from typing import Iterable, Optional

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar

from .._qt.utils import get_viewer_instance


def get_pbar():
    pbar = ProgressBar()
    viewer_instance = get_viewer_instance()
    viewer_instance.activityDock.widget().layout.addLayout(pbar.layout)

    return pbar


class progress:
    def __init__(
        self, iterable: Optional[Iterable] = None, total: Optional[int] = None
    ) -> None:
        self._iterable = iterable
        self._pbar = get_pbar()

        if iterable is not None:  # iterator takes priority over total
            try:
                self._total = len(iterable)
            except TypeError:  # generator (total needed)
                self._total = total if total is not None else 0
        else:
            if total is not None:
                self._total = total
            else:
                self._total = 0  # indeterminate bar

        self._pbar.set_total(self._total)

    def __iter__(self):
        for n, i in enumerate(self._iterable):
            self._pbar.set_value(n)
            yield i


class ProgressBar:
    def __init__(self) -> None:
        self.pbar = QProgressBar()
        self.label = QLabel("Test Label")

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pbar)
        self.layout = layout

    def set_total(self, total):
        self.pbar.setRange(0, total)

    def set_value(self, value):
        self.pbar.setValue(value)

    def set_description(self, desc):
        pass
