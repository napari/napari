"""The Window class is the primary entry to the napari GUI.

Currently, this module is just a stub file that will simply pass through the
:class:`napari._qt.qt_main_window.Window` class.  In the future, this module
could serve to define a window Protocol that a backend would need to implement
to server as a graphical user interface for napari.
"""

__all__ = ['Window']

from typing import Optional, Tuple

from napari.utils.translations import trans

try:
    from napari._qt import Window

except ImportError as e:
    err = e

    class Window:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self):
            pass

        def __getattr__(self, name):
            raise type(err)(
                trans._(
                    "An error occured when importing Qt dependencies.  Cannot show napari window.  See cause above",
                )
            ) from err

        def play_dim(
            self,
            axis: int = 0,
            fps: Optional[float] = None,
            loop_mode: Optional[str] = None,
            frame_range: Optional[Tuple[int, int]] = None,
        ):
            """
            Playback one dimension.

            Parameters
            ----------
            axis : int
                Index of axis to play
            fps : float
                Frames per second for playback.  Negative values will play in
                reverse.  fps == 0 will stop the animation. The view is not
                guaranteed to keep up with the requested fps, and may drop frames
                at higher fps.
            loop_mode : str
                Mode for animation playback.  Must be one of the following options:
                    "once": Animation will stop once movie reaches the
                        max frame (if fps > 0) or the first frame (if fps < 0).
                    "loop":  Movie will return to the first frame
                        after reaching the last frame, looping until stopped.
                    "back_and_forth":  Movie will loop back and forth until
                        stopped
            frame_range : tuple | list
                If specified, will constrain animation to loop [first, last] frames
            """

        def stop_dim(self):
            """Stop playback of dimension."""
