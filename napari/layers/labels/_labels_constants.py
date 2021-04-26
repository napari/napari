import sys
from collections import OrderedDict
from enum import auto

from ...utils.misc import StringEnum
from ...utils.translations import trans


class Mode(StringEnum):
    """MODE: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    In PICK mode the cursor functions like a color picker, setting the
    clicked on label to be the current label. If the background is picked it
    will select the background label `0`.

    In PAINT mode the cursor functions like a paint brush changing any pixels
    it brushes over to the current label. If the background label `0` is
    selected than any pixels will be changed to background and this tool
    functions like an eraser. The size and shape of the cursor can be adjusted
    in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace all
    pixels of that label or just those that are contiguous with the clicked on
    pixel. If the background label `0` is selected than any pixels will be
    changed to background and this tool functions like an eraser.

    In ERASE mode the cursor functions similarly to PAINT mode, but to paint
    with background label, which effectively removes the label.
    """

    PAN_ZOOM = auto()
    PICK = auto()
    PAINT = auto()
    FILL = auto()
    ERASE = auto()


class LabelColorMode(StringEnum):
    """
    LabelColorMode: Labelling Color setting mode.

    AUTO (default) allows color to be set via a hash function with a seed.

    DIRECT allows color of each label to be set directly by a color dictionary.

    SELECTED allows only selected labels to be visible
    """

    AUTO = auto()
    DIRECT = auto()


class LabelBrushShape(StringEnum):
    """
    LabelBrushShape: Labelling brush shape.

    CIRCLE (default) uses circle paintbrush (case insensitive).

    SQUARE uses square paintbrush (case insensitive).
    """

    CIRCLE = auto()
    SQUARE = auto()


BACKSPACE = 'delete' if sys.platform == 'darwin' else 'backspace'
LABEL_BRUSH_SHAPE_TRANSLATIONS = OrderedDict(
    [
        (LabelBrushShape.CIRCLE, trans._("circle")),
        (LabelBrushShape.SQUARE, trans._("square")),
    ]
)
LABEL_COLOR_MODE_TRANSLATIONS = OrderedDict(
    [
        (LabelColorMode.AUTO, trans._("auto")),
        (LabelColorMode.DIRECT, trans._("direct")),
    ]
)
