import numpy as np

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import EmitterGroup
from ._viewer_constants import Position


class ScaleBar:
    """Scale bar indicating size in world coordinates.

    Attributes
    ----------
    background_color : np.ndarray
        Background color of canvas. If scale bar is not colored
        then it has the color opposite of this color.
    colored : bool
        If scale bar are colored or not. If colored then
        default color is magenta. If not colored than
        scale bar color is the opposite of the canvas
        background.
    events : EmitterGroup
        Event emitter group
    position : str
        Position of the scale bar in the canvas. Must be one of
        'top left', 'top right', 'bottom right', 'bottom left'.
        Default value is 'bottom right'.
    ticks : bool
        If scale bar has ticks at ends or not.
    visible : bool
        If scale bar is visible or not.
    """

    def __init__(self):

        # Events:
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            visible=None,
            colored=None,
            position=None,
            ticks=None,
        )
        self._visible = False
        self._colored = False
        self._background_color = np.array([1, 1, 1])
        self._ticks = True
        self._position = Position('bottom_right')

    @property
    def visible(self):
        """bool: If scale bar is visible or not."""
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible
        self.events.visible()

    @property
    def colored(self):
        """bool: If scale bar is colored or not."""
        return self._colored

    @colored.setter
    def colored(self, colored):
        self._colored = colored
        self.events.colored()

    @property
    def background_color(self):
        """np.ndarray: RGBA color."""
        return self._background_color

    @background_color.setter
    def background_color(self, background_color):
        self._background_color = transform_color(background_color)[0]
        self.events.colored()

    @property
    def ticks(self):
        """bool: If scale bar has ticks or not."""
        return self._ticks

    @ticks.setter
    def ticks(self, ticks):
        self._ticks = ticks
        self.events.ticks()

    @property
    def position(self):
        """Position: Location of scale bar in canvas.

        Postion.TOP_LEFT
             Top left of the canvas.
        Postion.TOP_RIGHT
             Top right of the canvas.
        Postion.BOTTOM_RIGHT
             Bottom right of the canvas.
        Postion.BOTTOM_LEFT
             Bottom left of the canvas.
        """
        return str(self._position)

    @position.setter
    def position(self, position):
        self._position = Position(position)
        self.events.position()
