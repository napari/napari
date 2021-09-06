from ..vendored.filters.clipping_planes import PlanesClipper


class ClippingPlanesMixin:
    """
    Mixin class that attaches clipping planes filters to the (sub)visuals
    and provides property getter and setter
    """

    def __init__(self, *args, **kwargs):
        self._clip_filter = PlanesClipper()
        super().__init__(*args, **kwargs)

        # get subvisuals if compound, or just self
        visuals = getattr(self, '_subvisuals', [self])
        for vis in visuals:
            vis.attach(self._clip_filter)

    @property
    def clipping_planes(self):
        return self._clip_filter.clipping_planes

    @clipping_planes.setter
    def clipping_planes(self, value):
        self._clip_filter.clipping_planes = value
