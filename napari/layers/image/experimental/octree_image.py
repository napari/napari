"""OctreeImage class.
"""
from ....utils.events import Event
from ..image import Image
from ._chunked_slice_data import ChunkedSliceData
from ._octree_image_slice import OctreeImageSlice

DEFAULT_TILE_SIZE = 256


class OctreeImage(Image):
    """OctreeImage layer.

    Experimental variant of Image that renders using an Octree.

    Intended to eventually replace Image.
    """

    def __init__(self, *args, **kwargs):
        self._tile_size = DEFAULT_TILE_SIZE
        self._octree_level = None
        self._data_corners = None
        super().__init__(*args, **kwargs)
        self.events.add(octree_level=Event, tile_size=Event)

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int) -> None:
        self._tile_size = tile_size
        self.events.tile_size()
        self._slice = None
        self.refresh()

    @property
    def octree_level(self):
        """Return the currently displayed octree level."""
        return self._octree_level

    @octree_level.setter
    def octree_level(self, level: int):
        """Set the octree level we should be displaying.

        Parameters
        ----------
        level : int
            Display this octree level.
        """
        assert 0 <= level < self.num_octree_levels
        self._octree_level = level
        self.events.octree_level()
        self.refresh()  # Create new slice with this level.

    @property
    def num_octree_levels(self) -> int:
        """Return the total number of octree levels."""
        return self._slice.num_octree_levels

    def _new_empty_slice(self):
        """Initialize the current slice to an empty image.
        """
        self._slice = OctreeImageSlice(
            self._get_empty_image(),
            self._raw_to_displayed,
            self.rgb,
            self._tile_size,
            self._octree_level,
            self._data_corners,
        )
        self._empty = True

    @property
    def intersection(self):
        """Chunks in the current slice which in currently in view."""
        return self._slice.intersection

    @property
    def view_chunks(self):
        """Chunks in the current slice which in currently in view."""
        return self._slice.view_chunks

    def _on_data_loaded(self, data: ChunkedSliceData, sync: bool) -> None:
        """The given data a was loaded, use it now."""
        super()._on_data_loaded(data, sync)
        self._octree_level = self._slice._octree_level

        # TODO_OCTREE: The first time _on_data_loaded() is called it's from
        # super().__init__() and the octree_level event has not been added
        # yet. So we check here. This will go away when fold OctreeImage
        # back into Image.
        has_event = hasattr(self.events, 'octree_level')

        if has_event:
            self.events.octree_level()

    def _update_draw(self, scale_factor, corner_pixels, shape_threshold):

        # If self._data_corners was not set yet, we have not been drawn
        # yet, and we need to refresh to draw ourselves for the first time.
        need_refresh = self._data_corners is None

        self._data_corners = self._transforms[1:].simplified.inverse(
            corner_pixels
        )
        super()._update_draw(scale_factor, corner_pixels, shape_threshold)

        if need_refresh:
            self.refresh()
