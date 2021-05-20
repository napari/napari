import inspect
from typing import Iterable, Optional

from tqdm import tqdm

from ..utils.translations import trans

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

    monitor_interval = 0  # set to 0 to disable the thread

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        kwargs = kwargs.copy()
        pbar_kwargs = {k: kwargs.pop(k) for k in set(kwargs) - _tqdm_kwargs}

        # get progress bar added to viewer
        try:
            from .widgets.qt_progress_bar import get_pbar  # noqa

            pbar = get_pbar(**pbar_kwargs)
        # if no qt we revert to standard tqdm
        except ImportError:
            pbar = None

        self.has_viewer = pbar is not None
        if self.has_viewer:
            kwargs['gui'] = True

        super().__init__(iterable, desc, total, *args, **kwargs)
        if not self.has_viewer:
            return

        self._pbar = pbar
        if self.total is not None:
            self._pbar.setRange(self.n, self.total)
            self._pbar._set_value(self.n)
        else:
            self._pbar.setRange(0, 0)
            self.total = 0

        if desc:
            self.set_description(desc)
        else:
            self.set_description(trans._("progress"))

    def display(self, msg: str = None, pos: int = None) -> None:
        """Update the display."""
        if not self.has_viewer:
            return super().display(msg=msg, pos=pos)

        if self.total != 0:
            etas = str(self).split('|')[-1]
            self._pbar._set_value(self.n)
            self._pbar._set_eta(etas)

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)
        if self.has_viewer:
            self._pbar._set_description(self.desc)

    def close(self):
        """Closes and deletes the progress bar widget"""
        if self.disable:
            return
        if self.has_viewer:
            self._pbar.close()
            self._pbar.deleteLater()
        super().close()


def progrange(*args, **kwargs):
    """Shorthand for `progress(range(*args), **kwargs)`.

    Adds tqdm based progress bar to napari viewer, if it
    exists, and returns the wrapped range object.

    Returns
    -------
    progress
        wrapped range object
    """
    return progress(range(*args), **kwargs)
