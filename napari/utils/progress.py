import inspect
from typing import Iterable

# from magicgui import tqdm
from magicgui.widgets import FunctionGui

# from ..viewer import Viewer

try:
    from tqdm import tqdm as _tqdm_std
except ImportError as e:  # pragma: no cover
    msg = f"{e}. To use napari with tqdm please `pip install tqdm`"
    raise type(e)(msg)


def _find_viewer(max_depth=6):
    """Traverse calling stack looking for a napari Viewer."""
    stck = inspect.stack()
    for finfo in stck[2:max_depth]:

        if finfo.filename.endswith("viewer.py"):
            obj = finfo.frame.f_locals.get("self")
            if isinstance(obj, FunctionGui):
                return obj
            return None  # pragma: no cover

    return None


_tqdm_kwargs = {
    p.name
    for p in inspect.signature(_tqdm_std.__init__).parameters.values()
    if p.kind is not inspect.Parameter.VAR_KEYWORD and p.name != "self"
}


class progress(_tqdm_std):
    """magicgui version of tqdm.

    See tqdm.tqdm API for valid args and kwargs: https://tqdm.github.io/docs/tqdm/

    Also, any keyword arguments to the :class:`magicgui.widgets.ProgressBar` widget
    are also accepted and will be passed to the ``ProgressBar``.

    Examples
    --------
    When used inside of a magicgui-decorated function, ``tqdm`` (and the
    ``trange`` shortcut function) will append a visible progress bar to the gui
    container.

    >>> @magicgui(call_button=True)
    ... def long_running(steps=10, delay=0.1):
    ...     for i in tqdm(range(steps)):
    ...         sleep(delay)

    nesting is also possible:

    >>> @magicgui(call_button=True)
    ... def long_running(steps=10, repeats=4, delay=0.1):
    ...     for r in trange(repeats):
    ...         for s in trange(steps):
    ...             sleep(delay)
    """

    disable: bool

    def __init__(self, iterable: Iterable = None, *args, **kwargs) -> None:
        kwargs = kwargs.copy()
        # pbar_kwargs = {k: kwargs.pop(k) for k in set(kwargs) - _tqdm_kwargs}
        self._mgui = _find_viewer()

        super().__init__(iterable, *args, **kwargs)

        # self.sp = lambda x: None  # no-op status printer, required for older tqdm compat
        # if self.disable:
        #     return
