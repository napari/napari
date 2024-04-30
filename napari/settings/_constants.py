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


class LabelDTypes(StrEnum):
    uint8 = 'uint8'
    int8 = 'int8'
    uint16 = 'uint16'
    int16 = 'int16'
    uint32 = 'uint32'
    int32 = 'int32'
    uint64 = 'uint64'
    int64 = 'int64'
    uint = 'uint'
    int = 'int'


class NewLabelsPolicy(StrEnum):
    follow_image_class = 'Follow image class'
    fit_in_ram = 'Fit in RAM'
    follow_class_with_fit = (
        'Follow image class but fallback to fit in RAM if needed'
    )
