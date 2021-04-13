from typing import Iterable, Optional

from PyQt5.QtWidgets import QProgressBar

from .._qt.utils import get_viewer_instance


def get_pbar():
    pbar = ProgressBar()
    viewer_instance = get_viewer_instance()
    viewer_instance.activityDock.widget().layout.addWidget(pbar.pbar)

    return pbar


class progress:
    def __init__(
        self, iterable: Optional[Iterable] = None, total: Optional[int] = None
    ) -> None:
        self._iterable = iterable
        self._pbar = get_pbar()

        if iterable:
            try:
                self._total = len(iterable)
            except TypeError:  # generator (total needed)
                self._total = total
        else:
            if total:
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

    def set_total(total):
        pass

    def set_value(value):
        pass

    def set_description(desc):
        pass
