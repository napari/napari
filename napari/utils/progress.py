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


def get_pbar(viewer_instance, **kwargs):
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

        kwargs = kwargs.copy()
        pbar_kwargs = {k: kwargs.pop(k) for k in set(kwargs) - _tqdm_kwargs}

        super().__init__(iterable, desc, total, *args, **kwargs)
        if viewer is None:
            return
        self.viewer = viewer

        self._pbar = get_pbar(viewer, **pbar_kwargs)
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

        eta_params = {
            k: self.format_dict[k]
            for k in [
                'n',
                'total',
                'elapsed',
                'unit',
                'unit_scale',
                'rate',
                'unit_divisor',
                'initial',
            ]
        }
        etas = self.format_time(**eta_params)

        self._pbar._set_value(self.n)
        self._pbar._set_eta(etas)
        QApplication.processEvents()

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)

        self._pbar._set_description(self.desc)

    @staticmethod
    def format_time(
        n,
        total,
        elapsed,
        unit='it',
        unit_scale=False,
        rate=None,
        unit_divisor=1000,
        initial=0,
    ):

        # sanity check: total
        if total and n >= (total + 0.5):  # allow float imprecision (#849)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= (
                    unit_scale  # by default rate = self.avg_dn / self.avg_dt
                )
            unit_scale = False

        elapsed_str = tqdm.format_interval(elapsed)
        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        rate_noinv_fmt = (
            (
                (format_sizeof(rate) if unit_scale else '{:5.2f}'.format(rate))
                if rate
                else '?'
            )
            + unit
            + '/s'
        )
        rate_inv_fmt = (
            (
                (
                    format_sizeof(inv_rate)
                    if unit_scale
                    else '{:5.2f}'.format(inv_rate)
                )
                if inv_rate
                else '?'
            )
            + 's/'
            + unit
        )
        rate_fmt = (
            rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt
        )

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = (
                format_sizeof(total, divisor=unit_divisor)
                if total is not None
                else '?'
            )
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm.format_interval(remaining) if rate else '?'

        bar_etas = ' {}/{} [{}<{}, {}]'.format(
            n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt
        )

        return bar_etas

    def hide(self):
        """Hide the progress bar"""
        self._pbar.hide()

    def show(self):
        """Show the progress bar"""
        self._pbar.show()

    def close(self):
        """Closes and deletes the progress bar widget"""
        super().close()
        self._pbar.close()


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
