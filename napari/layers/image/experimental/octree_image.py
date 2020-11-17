"""OctreeImage class.
"""
from typing import List

import numpy as np

from ....components.experimental.chunk import ChunkRequest, chunk_loader
from ....utils.events import Event
from ..image import Image
from ._chunked_slice_data import ChunkedSliceData
from ._octree_multiscale_slice import OctreeMultiscaleSlice
from .octree_intersection import OctreeIntersection
from .octree_level import OctreeLevelInfo
from .octree_util import ImageConfig, OctreeChunk, OctreeChunkKey

DEFAULT_TILE_SIZE = 64

frame_num = 0


class OctreeImage(Image):
    """OctreeImage layer.

    Experimental variant of Image that renders using an Octree.

    Intended to eventually replace Image.
    """

    def __init__(self, *args, **kwargs):
        self.count = 0
        self._tile_size = DEFAULT_TILE_SIZE

        # Is this the same as Image._data_level? Which should we use?
        self._octree_level = None

        self._corners_2d = None
        self._auto_level = True
        self._track_view = True
        self._slice = None

        self.show_grid = True  # Get/set directly.

        # Temporary to implement a disabled cache.
        self._last_visible_set = set()

        super().__init__(*args, **kwargs)
        self.events.add(auto_level=Event, octree_level=Event, tile_size=Event)

    def _get_value(self):
        """Override Image._get_value()."""
        return (0, (0, 0))  # Fake for now until have octree version.

    @property
    def loaded(self):
        """Has the data for this layer been loaded yet."""
        # TODO_OCTREE: what here?
        return True

    @property
    def _empty(self) -> bool:
        return False  # TODO_OCTREE: what here?

    def _update_thumbnail(self):
        # TODO_OCTREE: replace Image._update_thumbnail with nothing for
        # the moment until we decide how to do thumbnail.
        pass

    @property
    def _data_view(self):
        """Viewable image for the current slice. (compatibility)"""
        # Override Image._data_view
        return np.zeros((64, 64, 3))  # fake: does octree need this?

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
    def image_config(self) -> ImageConfig:
        """Return information about the current octree.

        Return
        ------
        ImageConfig
            Basic image configuration.
        """
        if self._slice is None:
            return None
        return self._slice.image_config

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
        return len(self.data) - 1  # Multiscale

    def _new_empty_slice(self) -> None:
        """Initialize the current slice to an empty image.

        Overides Image._new_empty_slice() and does nothing because we don't
        need an empty slice. We create self._slice when
        self._set_view_slice() is called.

        The empty slice was needed to satisfy the old VispyImageLayer that
        used a single ImageVisual. But OctreeImage is drawn with
        VispyTiledImageVisual. It does not need an empty image. It gets
        chunks from our self.visible_chunks property, and it will just draw
        nothing if that returns an empty list.

        When OctreeImage become the only image class, this can go away.
        """

    @property
    def visible_chunks(self) -> List[OctreeChunk]:
        """Chunks in the current slice which in currently in view."""
        # This will be None if we have not been drawn yet.
        if self._slice is None or self._corners_2d is None:
            return []

        auto_level = self.auto_level and self.track_view

        if self._slice is None:
            return []

        chunks = self._slice.get_visible_chunks(self._corners_2d, auto_level)
        global frame_num
        print(
            f"OctreeImage.visible_chunks: frame={frame_num} num_chunks={len(chunks)}"
        )
        frame_num += 1

        visible_set = set(octree_chunk.key for octree_chunk in chunks)

        # Remove any chunks from our self._last_visible set which are no
        # longer in view.
        for key in list(self._last_visible_set):
            if key not in visible_set:
                self._last_visible_set.remove(key)

        # If we switched to a new octree level, update our currently shown level.
        slice_level = self._slice.octree_level
        if self._octree_level != slice_level:
            self._octree_level = slice_level
            self.events.octree_level()

        def _print(i, count, label, octree_chunk):
            print(f"Visible Chunk: {i} of {count} -> {label}: {octree_chunk}")

        # Visible chunks are ones that are already loaded or that we are
        # able to load synchronously. Perhaps in cache, etc.
        # visible_chunks = [
        #    octree_chunk
        #    for octree_chunk in chunks
        #    if not octree_chunk.needs_load or self._load_chunk(octree_chunk)
        # ]
        visible_chunks = []  # TODO_OCTREE combine list/set
        visible_set = set()
        for i, octree_chunk in enumerate(chunks):

            if not chunk_loader.cache.enabled:
                new_in_view = octree_chunk.key not in self._last_visible_set
                if new_in_view and octree_chunk.in_memory:
                    # Not using cache, so if this chunk just came into view
                    # clear it out, so it gets reloaded.
                    octree_chunk.clear()

            if octree_chunk.in_memory:
                # The chunk is fully in memory, we can view it right away.
                _print(i, len(chunks), "ALREADY LOADED", octree_chunk)
                visible_chunks.append(octree_chunk)
                visible_set.add(octree_chunk.key)
            elif octree_chunk.loading:
                # The chunk is being loaded, do not view it yet.
                _print(i, len(chunks), "LOADING:", octree_chunk)
            else:
                # The chunk is not in memory and is not being loaded, so
                # we are going to loaded it.
                if self._load_chunk(octree_chunk):
                    # The chunk was loaded synchronously. Either it hit the
                    # cache, or it's fast-loading data. We can draw it now.
                    _print(i, len(chunks), "SYNC LOAD", octree_chunk)
                    visible_chunks.append(octree_chunk)
                    visible_set.add(octree_chunk.key)
                else:
                    # An async load was initiated, sometime later our
                    # self._on_chunk_loaded method will be called.
                    _print(i, len(chunks), "ASYNC LOAD", octree_chunk)

        # Update our _last_visible_set with what is in view.
        for octree_chunk in chunks:
            self._last_visible_set.add(octree_chunk.key)

        return visible_chunks

    def _load_chunk(self, octree_chunk: OctreeChunk) -> None:

        indices = np.array(self._slice_indices)
        key = OctreeChunkKey(self, indices, octree_chunk.location)

        chunks = {'data': octree_chunk.data}

        octree_chunk.loading = True

        # Create the ChunkRequest and load it with the ChunkLoader.
        request = chunk_loader.create_request(self, key, chunks)

        satisfied_request = chunk_loader.load_chunk(request)

        if satisfied_request is None:
            return False  # Load was async.

        # Load was sync so we can insert the data into the octree
        # and we will draw it this frame.
        octree_chunk.data = satisfied_request.chunks.get('data')
        return True

    def _on_data_loaded(self, data: ChunkedSliceData, sync: bool) -> None:
        """The given data a was loaded, use it now."""

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

    def _outside_data_range(self, indices) -> bool:
        """Return True if requested slice is outside of data range.

        Return
        ------
        bool
            True if requested slice is outside data range.
        """

        extent = self._extent_data
        not_disp = self._dims.not_displayed

        return np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        )

    def _set_view_slice(self):
        """Set the view given the indices to slice with.

        This replaces Image._set_view_slice() entirely. The hope is eventually
        this class OctreeImage becomes Image. And the non-tiled multiscale
        logic in Image._set_view_slice goes away entirely.
        """
        if self._slice is not None:  # bail as a test
            return
        indices = np.array(self._slice_indices)
        if self._outside_data_range(indices):
            return

        rand_loc = 0
        rand_scale = 0
        image_config = ImageConfig.create(
            self.data[0].shape, self._tile_size, rand_loc, rand_scale
        )

        if self._slice is None:
            self._slice = OctreeMultiscaleSlice(
                self.data, image_config, self._raw_to_displayed
            )

    def on_chunk_loaded(self, request: ChunkRequest) -> None:
        """An asynchronous ChunkRequest was loaded.

        Override Image.on_chunk_loaded() fully.

        Parameters
        ----------
        request : ChunkRequest
            This request was loaded.
        """
        if self._slice.on_chunk_loaded(request):
            # Tell the visual to redraw with this new chunk.
            self.events.loaded()
