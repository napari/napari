from enum import auto

from napari.utils.compat import StrEnum
from napari.utils.misc import StringEnum


class LoopMode(StringEnum):
    """Looping mode for animating an axis.

    LoopMode.ONCE
        Animation will stop once movie reaches the max frame (if fps > 0) or
        the first frame (if fps < 0).
    LoopMode.LOOP
        Movie will return to the first frame after reaching the last frame,
        looping continuously until stopped.
    LoopMode.BACK_AND_FORTH
        Movie will loop continuously until stopped, reversing direction when
        the maximum or minimum frame has been reached.
    """

    ONCE = auto()
    LOOP = auto()
    BACK_AND_FORTH = auto()


class BrushSizeOnMouseModifiers(StrEnum):
    ALT = 'Alt'
    CTRL = 'Control'
    CTRL_ALT = 'Control+Alt'
    CTRL_SHIFT = 'Control+Shift'
    DISABLED = 'Disabled'  # a non-existent modifier that is never activated
