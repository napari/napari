import builtins
import sys
from enum import Enum, EnumMeta, auto

import numpy as np


class StringEnumMeta(EnumMeta):
    def __getitem__(self, item):
        """ set the item name case to uppercase for name lookup
        """
        if isinstance(item, str):
            item = item.upper()

        return super().__getitem__(item)

    def __call__(
        cls,
        value,
        names=None,
        *,
        module=None,
        qualname=None,
        type=None,
        start=1,
    ):
        """ set the item value case to lowercase for value lookup
        """
        # simple value lookup
        if names is None:
            if isinstance(value, str):
                return super().__call__(value.lower())
            elif isinstance(value, cls):
                return value
            else:
                raise ValueError(
                    f'{cls} may only be called with a `str`'
                    f' or an instance of {cls}: got {builtins.type(value)}'
                )

        # otherwise create new Enum class
        return cls._create_(
            value,
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )

    def keys(self):
        return list(map(str, self))


class StringEnum(Enum, metaclass=StringEnumMeta):
    def _generate_next_value_(name, start, count, last_values):
        """ autonaming function assigns each value its own name as a value
        """
        return name.lower()

    def __str__(self):
        """String representation: The string method returns the lowercase
        string of the Enum name
        """
        return self.value


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


class Interpolation(StringEnum):
    """INTERPOLATION: Vispy interpolation mode.

    The spatial filters used for interpolation are from vispy's
    spatial filters. The filters are built in the file below:

    https://github.com/vispy/vispy/blob/master/vispy/glsl/build-spatial-filters.py
    """

    BESSEL = auto()
    BICUBIC = auto()
    BILINEAR = auto()
    BLACKMAN = auto()
    CATROM = auto()
    GAUSSIAN = auto()
    HAMMING = auto()
    HANNING = auto()
    HERMITE = auto()
    KAISER = auto()
    LANCZOS = auto()
    MITCHELL = auto()
    NEAREST = auto()
    SPLINE16 = auto()
    SPLINE36 = auto()


class Interpolation3D(StringEnum):
    """INTERPOLATION: Vispy interpolation mode for volume rendering."""

    LINEAR = auto()
    NEAREST = auto()


class Rendering(StringEnum):
    """Rendering: Rendering mode for the layer.

    Selects a preset rendering mode in vispy
            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maximum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and attenuate values based on integral of encountered values,
              display the maximum value that was encountered after attenuation.
              This will make nearer objects appear more prominent.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    ISO = auto()
    MIP = auto()
    ATTENUATED_MIP = auto()


class Blending(StringEnum):
    """BLENDING: Blending mode for the layer.

    Selects a preset blending mode in vispy that determines how
            RGB and alpha values get mixed.
            Blending.OPAQUE
                Allows for only the top layer to be visible and corresponds to
                depth_test=True, cull_face=False, blend=False.
            Blending.TRANSLUCENT
                Allows for multiple layers to be blended with different opacity
                and corresponds to depth_test=True, cull_face=False,
                blend=True, blend_func=('src_alpha', 'one_minus_src_alpha').
            Blending.ADDITIVE
                Allows for multiple layers to be blended together with
                different colors and opacity. Useful for creating overlays. It
                corresponds to depth_test=False, cull_face=False, blend=True,
                blend_func=('src_alpha', 'one').
    """

    TRANSLUCENT = auto()
    ADDITIVE = auto()
    OPAQUE = auto()


class LabelsMode(StringEnum):
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


class DimsMode(Enum):
    POINT = 0
    INTERVAL = 1


class TextMode(StringEnum):
    """
    TextMode: Text setting mode.

    NONE (default mode) no text is displayed

    PROPERTY the text value is a property value

    FORMATTED allows text to be set with an f-string like syntax
    """

    NONE = auto()
    PROPERTY = auto()
    FORMATTED = auto()


class Anchor(StringEnum):
    """
    Anchor: The anchor position for text

    CENTER The text origin is centered on the layer item bounding box.

    UPPER_LEFT The text origin is on the upper left corner of the bounding box
    UPPER_RIGHT The text origin is on the upper right corner of the bounding box
    LOWER_LEFT The text origin is on the lower left corner of the bounding box
    LOWER_RIGHT The text origin is on the lower right corner of the bounding box
    """

    CENTER = auto()
    UPPER_LEFT = auto()
    UPPER_RIGHT = auto()
    LOWER_LEFT = auto()
    LOWER_RIGHT = auto()


class PointsMode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    ADD allows points to be added by clicking

    SELECT allows the user to select points by clicking on them
    """

    ADD = auto()
    SELECT = auto()
    PAN_ZOOM = auto()


class Symbol(Enum):
    """Symbol: Valid symbol/marker types for the Points layer.
    The string method returns the valid vispy string.

    """

    ARROW = 'arrow'
    CLOBBER = 'clobber'
    CROSS = 'cross'
    DIAMOND = 'diamond'
    DISC = 'disc'
    HBAR = 'hbar'
    RING = 'ring'
    SQUARE = 'square'
    STAR = 'star'
    TAILED_ARROW = 'tailed_arrow'
    TRIANGLE_DOWN = 'triangle_down'
    TRIANGLE_UP = 'triangle_up'
    VBAR = 'vbar'
    X = 'x'

    def __str__(self):
        """String representation: The string method returns the
        valid vispy symbol string for the Markers visual.
        """
        return self.value


# Mapping of symbol alias names to the deduplicated name
SYMBOL_ALIAS = {
    'o': Symbol.DISC,
    '*': Symbol.STAR,
    '+': Symbol.CROSS,
    '-': Symbol.HBAR,
    '->': Symbol.TAILED_ARROW,
    '>': Symbol.ARROW,
    '^': Symbol.TRIANGLE_UP,
    'v': Symbol.TRIANGLE_DOWN,
    's': Symbol.SQUARE,
    '|': Symbol.VBAR,
}


class ShapesMode(StringEnum):
    """Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    The SELECT mode allows for entire shapes to be selected, moved and
    resized.

    The DIRECT mode allows for shapes to be selected and their individual
    vertices to be moved.

    The VERTEX_INSERT and VERTEX_REMOVE modes allow for individual
    vertices either to be added to or removed from shapes that are already
    selected. Note that shapes cannot be selected in this mode.

    The ADD_RECTANGLE, ADD_ELLIPSE, ADD_LINE, ADD_PATH, and ADD_POLYGON
    modes all allow for their corresponding shape type to be added.
    """

    PAN_ZOOM = auto()
    SELECT = auto()
    DIRECT = auto()
    ADD_RECTANGLE = auto()
    ADD_ELLIPSE = auto()
    ADD_LINE = auto()
    ADD_PATH = auto()
    ADD_POLYGON = auto()
    VERTEX_INSERT = auto()
    VERTEX_REMOVE = auto()


class ColorMode(StringEnum):
    """
    ColorMode: Color setting mode.

    DIRECT (default mode) allows each shape to be set arbitrarily

    CYCLE allows the color to be set via a color cycle over an attribute

    COLORMAP allows color to be set via a color map over an attribute
    """

    DIRECT = auto()
    CYCLE = auto()
    COLORMAP = auto()


DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Box:
    """Box: Constants associated with the vertices of the interaction box
    """

    WITH_HANDLE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    LINE_HANDLE = [7, 6, 4, 2, 0, 7, 8]
    LINE = [0, 2, 4, 6, 0]
    TOP_LEFT = 0
    TOP_CENTER = 7
    LEFT_CENTER = 1
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 2
    CENTER = 8
    HANDLE = 9
    LEN = 8


BACKSPACE = 'delete' if sys.platform == 'darwin' else 'backspace'


class ShapeType(StringEnum):
    """ShapeType: Valid shape type."""

    RECTANGLE = auto()
    ELLIPSE = auto()
    LINE = auto()
    PATH = auto()
    POLYGON = auto()
