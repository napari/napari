import inspect
from typing import Iterable, Optional

from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)
from tqdm import tqdm

from .._qt.utils import get_viewer_instance


def get_pbar(viewer_instance, **kwargs):
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
    pbar = ProgressBar(**kwargs)
    viewer_instance.activityDock.widget().layout.addWidget(pbar)

    return pbar


def get_calling_function_name(max_depth: int):
    """Inspect stack up to max_depth and return first function name outside of progress.py"""
    for finfo in inspect.stack()[2:max_depth]:
        if not finfo.filename.endswith("progress.py"):
            return finfo.function

    return None


_tqdm_kwargs = {
    p.name
    for p in inspect.signature(tqdm.__init__).parameters.values()
    if p.kind is not inspect.Parameter.VAR_KEYWORD and p.name != "self"
}


class progress(tqdm):
    """This class inherits from tqdm and provides an interface for
    progress bars in the napari viewer. Progress bars can be created
    directly by wrapping an iterable or by providing a total number
    of expected updates.

    See tqdm.tqdm API for valid args and kwargs:
    https://tqdm.github.io/docs/tqdm/

    Also, any keyword arguments to the :class:`ProgressBar` `QWidget`
    are also accepted and will be passed to the ``ProgressBar``.

    Examples
    --------

    >>> def long_running(steps=10, delay=0.1):
    ...     for i in progress(range(steps)):
    ...         sleep(delay)

    it can also be used as a context manager:

    >>> def long_running(steps=10, repeats=4, delay=0.1):
    ...     with progress(range(steps)) as pbr:
    ...         for i in pbr:
    ...             sleep(delay)

    or equivalently, using the `progrange` shorthand
    ...     with progrange(steps) as pbr:
    ...         for i in pbr:
    ...             sleep(delay)

    For manual updates:

    >>> def manual_updates(total):
    ...     pbr = progress(total=total)
    ...     sleep(10)
    ...     pbr.set_description("Step 1 Complete")
    ...     pbr.update(1)
    ...     # must call pbr.close() when using outside for loop
    ...     # or context manager
    ...     pbr.close()

    """

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
        self.has_viewer = viewer is not None
        if self.has_viewer:
            kwargs['gui'] = True

        kwargs = kwargs.copy()
        pbar_kwargs = {k: kwargs.pop(k) for k in set(kwargs) - _tqdm_kwargs}

        super().__init__(iterable, desc, total, *args, **kwargs)
        if not self.has_viewer:
            return

        self._pbar = get_pbar(viewer, **pbar_kwargs)
        if self.total is not None:
            self._pbar.setRange(self.n, self.total)
            self._pbar._set_value(self.n)
        else:
            self._pbar.setRange(0, 0)

        if desc:
            self.set_description(desc)
        else:
            desc = get_calling_function_name(max_depth=5)
            if desc:
                self.set_description(desc)
            else:
                # TODO: pick a better default
                self.set_description("Progress Bar")

        self.show()
        QApplication.processEvents()

    def display(self, msg: str = None, pos: int = None) -> None:
        """Update the display."""
        if not self.has_viewer:
            return super().display(msg=msg, pos=pos)

        etas = self.get_formatted_etas()
        self._pbar._set_value(self.n)
        self._pbar._set_eta(etas)
        QApplication.processEvents()

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)
        if self.has_viewer:
            self._pbar._set_description(self.desc)

    def get_formatted_etas(self):
        """Strip ascii formatted progress bar and return formatted etas.

        Returns
        -------
        str
            formatted eta and remaining iteration details
        """
        formatted_bar = str(self)
        if "|" in formatted_bar:
            return formatted_bar.split("|")[-1]
        else:
            return ""

    def hide(self):
        """Hide the progress bar"""
        if self.has_viewer:
            self._pbar.hide()

    def show(self):
        """Show the progress bar"""
        if self.has_viewer:
            self._pbar.show()

    def close(self):
        """Closes and deletes the progress bar widget"""
        if self.disable:
            return
        if self.has_viewer:
            self._pbar.close()
        super().close()


def progrange(*args, **kwargs):
    return progress(range(*args), **kwargs)


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

    def _get_value(self):
        return self.pbar.value()

    def _set_description(self, desc):
        self.description_label.setText(desc)

    def _set_eta(self, eta):
        self.eta_label.setText(eta)
