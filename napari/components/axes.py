import numpy as np

from ..utils.colormaps.standardize_color import transform_color
from ..utils.events import EmitterGroup


class Axes:
    """Axes indicating world coordinate origin and orientation.

    Attributes
    ----------
    events : EmitterGroup
        Event emitter group
    visible : bool
        If axes are visible or not.
    labels : bool
        If axes labels are visible or not. Not the actual
        axes labels are stored in `viewer.dims.axes_labels`.
    colored : bool
        If axes are colored or not. If colored then default
        coloring is x=cyan, y=yellow, z=magenta. If not
        colored than axes are the color opposite of
        the canvas background.
    dashed : bool
        If axes are dashed or not. If not dashed then
        all the axes are solid. If dashed then x=solid,
        y=dashed, z=dotted.
    background_color : np.ndarray
        Background color of canvas. If axes are not colored
        then they have the color opposite of this color.
    arrows : bool
        If axes have arrowheads or not.
    """

    def __init__(self):

        # Events:
        self.events = EmitterGroup(
            source=self,
            auto_connect=True,
            visible=None,
            colored=None,
            dashed=None,
            arrows=None,
            labels=None,
        )
        self._visible = False
        self._labels = True
        self._colored = True
        self._background_color = np.array([1, 1, 1])
        self._dashed = False
        self._arrows = True

    @property
    def visible(self):
        """bool: If axes are visible or not."""
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible
        self.events.visible()

    @property
    def labels(self):
        """bool: If axes labels are visible or not."""
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        self.events.labels()

    @property
    def colored(self):
        """bool: If axes are colored or not."""
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
    def dashed(self):
        """bool: If axes are dashed or not."""
        return self._dashed

    @dashed.setter
    def dashed(self, dashed):
        self._dashed = dashed
        self.events.dashed()

    @property
    def arrows(self):
        """bool: If axes have arrowheads or not."""
        return self._arrows

    @arrows.setter
    def arrows(self, arrows):
        self._arrows = arrows
        self.events.arrows()
