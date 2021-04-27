"""The Window class is the primary entry to the napari GUI.

Currently, this module is just a stub file that will simply pass through the
:class:`napari._qt.qt_main_window.Window` class.  In the future, this module
could serve to define a window Protocol that a backend would need to implement
to server as a graphical user interface for napari.
"""

__all__ = ['Window']

try:
    from ._qt import Window
    from .utils.translations import trans

except ImportError:

    class Window:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def __getattr__(self, name):
            raise ImportError(
                trans._("could not import `qtpy`. Cannot show napari window.")
            )
