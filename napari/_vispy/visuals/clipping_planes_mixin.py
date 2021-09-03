from ..vendored.filters.clipping_planes import PlanesClipper


class ClippingPlanesMixin:
    def __init__(self):
        self._clip_filter = PlanesClipper()
        super().__init__()

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
