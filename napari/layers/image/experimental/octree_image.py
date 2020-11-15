"""OctreeImage class.
"""
from typing import List

from ....utils.events import Event
from ..image import Image
from ._chunked_slice_data import ChunkedSliceData
from ._octree_image_slice import OctreeImageSlice
from ._octree_multiscale_slice import OctreeMultiscaleSlice
from .octree_intersection import OctreeIntersection
from .octree_level import OctreeLevelInfo
from .octree_util import ChunkData, OctreeInfo

DEFAULT_TILE_SIZE = 64


class OctreeImage(Image):
    """OctreeImage layer.

    Experimental variant of Image that renders using an Octree.

    Intended to eventually replace Image.
    """

    def __init__(self, *args, **kwargs):
        self._tile_size = DEFAULT_TILE_SIZE

        # Is this the same as Image._data_level? Which should we use?
        self._octree_level = None

        self._corners_2d = None
        self._auto_level = True
        self._track_view = True

        self.show_grid = True  # Get/set directly.

        super().__init__(*args, **kwargs)
        self.events.add(auto_level=Event, octree_level=Event, tile_size=Event)

    @property
    def track_view(self) -> bool:
        """Return True if we changing what's dispays as the view changes.

        Return
        ------
        bool
            True if we are tracking the current view.
        """
        return self._track_view

    @track_view.setter
    def track_view(self, value: bool) -> None:
        """Set whether we are tracking the current view.

        Parameters
        ----------
        value : bool
            True if we should track the current view.
        """
        self._track_view = value

    @property
    def tile_size(self) -> int:
        """Return the edge length of single tile, for example 256.

        Return
        ------
        int
            The edge length of a single tile.
        """
        return self._tile_size

    @property
    def tile_shape(self) -> tuple:
        """Return the shape of a single tile, for example 256x256x3.

        Return
        ------
        tuple
            The shape of a single tile.
        """
        # TODO_OCTREE: Must be an easier way to get this shape based on
        # information already stored in Image class?
        if self.multiscale:
            init_shape = self.data[0].shape
        else:
            init_shape = self.data.shape

        tile_shape = (self.tile_size, self.tile_size)

        if self.rgb:
            # Add the color dimension (usually 3 or 4)
            tile_shape += (init_shape[-1],)

        return tile_shape

    @tile_size.setter
    def tile_size(self, tile_size: int) -> None:
        self._tile_size = tile_size
        self.events.tile_size()
        self._slice = None
        self.refresh()

    @property
    def octree_info(self) -> OctreeInfo:
        """Return information about the current octree.

        Return
        ------
        OctreeInfo
            Information about the current octree.
        """
        if self._slice is None:
            return None
        return self._slice.octree_info

    @property
    def octree_level_info(self) -> OctreeLevelInfo:
        """Return information about the current level of the current octree.

        Returns
        -------
        OctreeLevelInfo
            Information about the current octree level.
        """
        if self._slice is None:
            return None
        return self._slice.octree_level_info

    @property
    def auto_level(self) -> bool:
        """Return True if we are computing the octree level automatically.

        When viewing the octree normally, auto_level is always True, but
        during debugging or other special situations it might be off.

        Returns
        -------
        bool
            True if we are computing the octree level automatically.
        """
        return self._auto_level

    @auto_level.setter
    def auto_level(self, value: bool) -> None:
        """Set whether we are choosing the octree level automatically.

        Parameters
        ----------
        value : bool
            True if we should determine the octree level automatically.
        """
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

        Overides Image._new_empty_slice so we can create slices that
        render using an octree.
        """
        if self.multiscale:
            self._slice = OctreeMultiscaleSlice()
        else:
            self._slice = OctreeImageSlice(
                self._get_empty_image(),
                self._raw_to_displayed,
                self.rgb,
                self._tile_size,
                self._octree_level,
            )
        self._empty = True

    @property
    def visible_chunks(self) -> List[ChunkData]:
        """Chunks in the current slice which in currently in view."""
        # This will be None if we have not been drawn yet.
        if self._corners_2d is None:
            return []

        auto_level = self.auto_level and self.track_view

        chunks = self._slice.get_visible_chunks(self._corners_2d, auto_level)
        self._octree_level = self._slice.octree_level
        self.events.octree_level()
        return chunks

    def _on_data_loaded(self, data: ChunkedSliceData, sync: bool) -> None:
        """The given data a was loaded, use it now."""
        super()._on_data_loaded(data, sync)
        self._octree_level = self._slice.octree_level

        # TODO_OCTREE: The first time _on_data_loaded() is called it's from
        # super().__init__() and the octree_level event has not been added
        # yet. So we check here. This will go away when fold OctreeImage
        # back into Image.
        has_event = hasattr(self.events, 'octree_level')

        if has_event:
            self.events.octree_level()

    def _update_draw(self, scale_factor, corner_pixels, shape_threshold):

        # Need refresh if have not been draw at all yet.
        need_refresh = self._corners_2d is None

        # Compute self._corners_2d which we use for intersections.
        data_corners = self._transforms[1:].simplified.inverse(corner_pixels)
        self._corners_2d = self._convert_to_corners_2d(data_corners)

        super()._update_draw(scale_factor, corner_pixels, shape_threshold)

        if need_refresh:
            self.refresh()

    def get_intersection(self) -> OctreeIntersection:
        """The the interesection between the current view and the octree.

        Returns
        -------
        OctreeIntersection
            The intersection between the current view and the octree.
        """
        if self._slice is None:
            return None

        return self._slice.get_intersection(self._corners_2d, self.auto_level)

    def _convert_to_corners_2d(self, data_corners):
        """
        Get data corners in 2d.
        """
        # TODO_OCTREE: This is placeholder. Need to handle dims correctly.
        if self.ndim == 2:
            return data_corners
        return data_corners[:, 1:3]
