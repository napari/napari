from __future__ import annotations

import os
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from napari._version import __version__
from napari.utils.notifications import notification_manager, show_warning
from napari.utils.translations import trans

if TYPE_CHECKING:
    from pathlib import Path

    from napari.viewer import Viewer

_SCRIPT_NAMESPACES: dict[str | Path, dict[str, Any]] = {}
# This is a global dictionary to store the namespace for scripts that are
# executed using drag and drop or the Python file reader.
# The content is a mapping from the script path to the namespace.
# The dict should not be overwritten but only modified, as
# it may be broken the execution of scripts that are already running.


def imsave(filename: str, data: np.ndarray):
    """Custom implementation of imsave to avoid skimage dependency.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    ext = os.path.splitext(filename)[1].lower()
    # If no file extension was specified, choose .png by default
    if ext == '':
        if (
            data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in {3, 4})
        ) and not np.issubdtype(data.dtype, np.floating):
            ext = '.png'
        else:
            ext = '.tif'
            filename = filename + ext
    # not all file types can handle float data
    if ext not in [
        '.tif',
        '.tiff',
        '.bsdf',
        '.im',
        '.lsm',
        '.npz',
        '.stk',
    ] and np.issubdtype(data.dtype, np.floating):
        show_warning(
            trans._(
                'Image was not saved, because image data is of dtype float.\nEither convert dtype or save as different file type (e.g. TIFF).'
            )
        )
        return
    # Save screenshot image data to output file
    if ext in ['.png']:
        imsave_png(filename, data)
    elif ext in ['.tif', '.tiff']:
        imsave_tiff(filename, data)
    else:
        import imageio.v3 as iio

        iio.imwrite(filename, data)  # for all other file extensions


def imsave_png(filename, data):
    """Save .png image to file

    PNG images created in napari have a digital watermark.
    The napari version info is embedded into the bytes of the PNG.

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    import imageio.v3 as iio
    import PIL.PngImagePlugin

    # Digital watermark, adds info about the napari version to the bytes of the PNG file
    pnginfo = PIL.PngImagePlugin.PngInfo()
    pnginfo.add_text(
        'Software', f'napari version {__version__} https://napari.org/'
    )
    iio.imwrite(
        filename,
        data,
        extension='.png',
        plugin='pillow',
        pnginfo=pnginfo,
    )


def imsave_tiff(filename, data):
    """Save .tiff image to file

    Parameters
    ----------
    filename : string
        The path to write the file to.
    data : np.ndarray
        The image data.
    """
    import tifffile

    if data.dtype == bool:
        tifffile.imwrite(filename, data)
    else:
        try:
            tifffile.imwrite(
                filename,
                data,
                # compression arg structure since tifffile 2022.7.28
                compression='zlib',
                compressionargs={'level': 1},
            )
        except struct.error:
            # regular tiffs don't support compressed data >4GB
            # in that case a struct.error is raised, and we write with the
            # bigtiff flag. (The flag is not on by default because it is
            # not as widely supported as normal tiffs.)
            tifffile.imwrite(
                filename,
                data,
                compression='zlib',
                compressionargs={'level': 1},
                bigtiff=True,
            )


def execute_python_code(code: str, script_path: str | Path = '') -> None:
    """Execute Python code in the current viewer's context.

    Store the executed cod variables in _SCRIPT_NAMESPACES dict

    Parameters
    ----------
    code: str
        python code to be executed
    script_path: str | Path
        Path to the script file from which the code is executed.
        Used to store the namespace in the _SCRIPT_NAMESPACES.
    """
    from napari.viewer import current_viewer

    with _patched_viewer_new(), _noop_napari_run():
        try:
            viewer = current_viewer()
            script_namespace = _SCRIPT_NAMESPACES.setdefault(script_path, {})
            # The `__name__` variable is storing the name of the module.
            # If a module is imported, it is set to the module name.
            # If a module is executed with `python -m ...` or
            # `python script.py` it is set to '__main__'.
            # If code is executed with `exec(code, namespace)` it is set to `builtins` if
            # `__name__` is not set in the namespace.
            # So we set it to `__main__` to execute `if __name__ == '__main__':` blocks
            script_namespace['__name__'] = '__main__'
            exec(code, script_namespace)
            _add_variables_to_viewer_console(
                _SCRIPT_NAMESPACES[script_path], viewer
            )
        except BaseException as e:  # noqa: BLE001
            notification_manager.receive_error(type(e), e, e.__traceback__)


@contextmanager
def _patched_viewer_new():
    """Context manager to patch the viewer's new method."""
    from napari.viewer import Viewer, current_viewer

    original_new = Viewer.__new__
    original_init = Viewer.__init__

    def patched_init(self, *args, **kwargs):
        Viewer.__init__ = original_init

    def patched_new(cls, *args, **kwargs):
        ndisplay = None
        if len(kwargs) == 1 and 'ndisplay' in kwargs:
            ndisplay = kwargs.pop('ndisplay')

        if not kwargs and not args:
            viewer = current_viewer()
            if ndisplay is not None:
                viewer.dims.ndisplay = ndisplay  # type: ignore
            if viewer is not None:
                Viewer.__new__ = original_new
                return viewer
        Viewer.__init__ = original_init
        return original_new(cls)

    Viewer.__new__ = patched_new
    Viewer.__init__ = patched_init
    try:
        yield
    finally:
        Viewer.__new__ = original_new
        Viewer.__init__ = original_init


@contextmanager
def _noop_napari_run():
    """Context manager to patch napari.run to always be a no-op.

    napari.run() executes the Qt event loop, *except* when napari
    is running in IPython and therefore IPython's Qt integration
    already has the event loop.

    When running a script by dragging-and-dropping onto a
    running napari Viewer, we already have an event loop, so we
    should not start a new nested loop, even though we are not
    in IPython.

    This context manager temporarily patches the IPython check
    to always return True, causing a fast exit from napari.run()
    without a new event loop.
    """
    from napari._qt import qt_event_loop

    original_ipython_check = qt_event_loop._ipython_has_eventloop

    def patched_ipython_check() -> bool:
        """A patched ipython_check that always returns True.

        napari's script running from drag-and-dropping a script
        into a viewer uses this patch to prevent nested event loops.
        """
        return True

    qt_event_loop._ipython_has_eventloop = patched_ipython_check
    try:
        yield
    finally:
        qt_event_loop._ipython_has_eventloop = original_ipython_check


def _filter_variables(variables: dict[str, Any]) -> dict[str, Any]:
    res = variables.copy()
    res.pop('viewer', None)
    res.pop(
        '__name__', None
    )  # Remove the __name__ variable to not affect the console
    return res


def _add_variables_to_viewer_console(
    variables: dict[str, Any], viewer: Viewer | None
) -> None:
    if viewer is None:
        return

    variables = _filter_variables(variables)

    if viewer.window._qt_viewer._console is None:
        viewer.window._qt_viewer.add_to_console_backlog(variables)
    else:
        console = viewer.window._qt_viewer._console
        console.push(variables)
