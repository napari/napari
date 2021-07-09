from time import time
from typing import Iterable, Optional

from napari._qt.widgets.qt_progress_bar import ProgressBar, ProgressBarGroup

from ..utils.translations import trans


class progress:
    """This class provides an interface for
    progress bars in the napari viewer. Progress bars can be created
    directly by wrapping an iterable or by providing a total number
    of expected updates.

    Any keyword arguments to the :class:`ProgressBar` `QWidget`
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
        step: Optional[int] = None,
        nest_under: Optional[ProgressBar] = None,
        *args,
        **kwargs,
    ) -> None:

        self.iterable = iterable
        self.n = 0

        if iterable is not None:  # iterator takes priority over total
            try:
                self.total = len(iterable)
            except TypeError:  # generator (total needed)
                self.total = total if total is not None else 0
        else:
            if total is not None:
                self.total = total
                self.step = step if step else 1
                self.iterable = range(0, total, self.step)
            else:
                self.total = 0
                self.step = 0

        # get progress bar added to viewer
        try:
            from .dialogs.activity_dialog import get_pbar

            pbar = get_pbar(self, nest_under=nest_under, **kwargs)
        except ImportError:
            pbar = None

        self._pbar = pbar
        if not self._pbar:
            raise TypeError(
                trans._(
                    "Cannot use progress object without an active napari viewer.",
                    deferred=True,
                )
            )

        if self.total is not None:
            self._pbar.setRange(self.n, self.total)
            self._pbar._set_value(self.n)
        else:
            self._pbar.setRange(0, 0)
            self.total = 0

        if desc:
            self.desc = desc
            self.set_description(desc)
        else:
            self.desc = "progress"
            self.set_description(trans._(self.desc))

        self.last_update_t = time()
        self.start_t = self.last_update_t

    def __iter__(self):
        iterable = self.iterable
        n = self.n
        try:
            for obj in iterable:
                yield obj

                n += 1
                self.update(1)
        finally:
            self.n = n
            self.close()

    def __len__(self):
        if self.iterable is None:
            return self.total
        elif hasattr(self.iterable, 'shape'):
            return self.iterable.shape[0]
        elif hasattr(self.iterable, '__len__'):
            return len(self.iterable)
        else:
            return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def increment(self):
        """Increment progress bar using current step"""
        if self._pbar:
            self._pbar._set_value(
                min(self.total, self._pbar._get_value() + self.step)
            )

    def decrement(self):
        """Decrement progress bar using current step"""
        if self._pbar:
            self._pbar._set_value(max(0, self._pbar._get_value() - self.step))

    def update(self, n):
        """Update progress bar with new value
        Parameters
        ----------
        n : int
            increment to add to internal iteration counter
        """
        self.n += n
        if self._pbar:
            if self.n > self.total:
                # exceeded total, become indeterminate
                self._pbar._set_total(0)
                self.n = 0
            else:
                self._pbar._set_value(self.n)
            if self.total != 0:
                cur_t = time()
                dt_total = cur_t - self.start_t
                avg_dt_iter = dt_total / self.n
                eta = (avg_dt_iter * self.total) - dt_total
                self._pbar._set_eta(
                    f"{self.n}/{self.total} [{eta:.2f}<{dt_total:.2f}, {avg_dt_iter:.2f}s/it]"
                )
                self.last_update_t = cur_t

    def increment_with_overflow(self):
        """Update if not exceeding total, else set indeterminate range."""
        if self.n == self.total:
            self.total = 0
            if self._pbar:
                self._pbar.setRange(0, 0)
        else:
            self.update(self.step)

    def set_description(self, desc):
        """Update progress bar description"""
        if self._pbar:
            self.desc = desc
            self._pbar._set_description(desc)

    def close(self):
        """Closes and deletes the progress bar widget"""
        if self._pbar:
            self.close_pbar()

    def close_pbar(self):
        if self._pbar:
            parent_widget = self._pbar.parent()
            self._pbar.close()
            self._pbar.deleteLater()
            if isinstance(parent_widget, ProgressBarGroup):
                pbar_children = [
                    child
                    for child in parent_widget.children()
                    if isinstance(child, ProgressBar)
                ]
                if not any(child.isVisible() for child in pbar_children):
                    parent_widget.close()
            self._pbar = None


def progrange(*args, **kwargs):
    """Shorthand for `progress(range(*args), **kwargs)`.

    Adds progress bar to napari viewer, if it
    exists, and returns the wrapped range object.

    Returns
    -------
    progress
        wrapped range object
    """
    return progress(range(*args), **kwargs)
