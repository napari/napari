from typing import List, Optional

from typing import Protocol
from vispy.visuals.filters import Filter
from vispy.visuals.filters.clipping_planes import PlanesClipper


class _PVisual(Protocol):
    """
    Type for vispy visuals that implement the attach method
    """

    _subvisuals: Optional[List['_PVisual']]

    def attach(self, filt: Filter, view=None):
        ...


class ClippingPlanesMixin:
    """
    Mixin class that attaches clipping planes filters to the (sub)visuals
    and provides property getter and setter
    """

    def __init__(self: _PVisual, *args, **kwargs):
        self._clip_filter = PlanesClipper()
        super().__init__(*args, **kwargs)

        self.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
