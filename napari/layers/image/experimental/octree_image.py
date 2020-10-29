"""OctreeImage class.
"""
from typing import List

from ....utils.events import Event
from ..image import Image
from ._chunked_slice_data import ChunkedSliceData
from ._octree_image_slice import OctreeImageSlice
from .octree_intersection import OctreeIntersection
from .octree_util import ChunkData, OctreeInfo, OctreeLevelInfo

DEFAULT_TILE_SIZE = 64


class OctreeImage(Image):
    """OctreeImage layer.

    Experimental variant of Image that renders using an Octree.

    Intended to eventually replace Image.
    """

    def __init__(self, *args, **kwargs):
        self._tile_size = DEFAULT_TILE_SIZE
        self._octree_level = None
        self._data_corners = None
        self._auto_level = True
        super().__init__(*args, **kwargs)
        self.events.add(auto_level=Event, octree_level=Event, tile_size=Event)

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
    def octree_info(self) -> OctreeInfo:
        if self._slice is None:
            return None
        else:
            return self._slice.octree_info

    @property
    def octree_level_info(self) -> OctreeLevelInfo:
        if self._slice is None:
            return None
        else:
            return self._slice.octree_level_info

    @property
    def auto_level(self) -> bool:
        return self._auto_level

    @auto_level.setter
    def auto_level(self, value: bool) -> None:
        self._auto_level = value
        self.events.auto_level()

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
        )
        self._empty = True

    @property
    def intersection(self):
        """Chunks in the current slice which in currently in view."""
        return self._slice.intersection

    @property
    def visible_chunks(self) -> List[ChunkData]:
        """Chunks in the current slice which in currently in view."""
        # This will be None if we have not been drawn yet.
        if self._data_corners is None:
            return []

        corners_2d = self._corners_2d(self._data_corners)
        chunks = self._slice.get_visible_chunks(corners_2d, self._auto_level)
        self._octree_level = self._slice._octree_level
        self.events.octree_level()
        return chunks

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

    def get_intersection(self, data_corners) -> OctreeIntersection:
        if self._slice is None:
            return None

        corners_2d = self._corners_2d(data_corners)
        return self._slice.get_intersection(corners_2d, self.auto_level)

    def _corners_2d(self, data_corners):
        """
        Get data corners in 2d.
        """
        # TODO_OCTREE: This is placeholder. Need to handle dims correctly.
        if self.ndim == 2:
            return data_corners
        else:
            return data_corners[:, 1:3]
