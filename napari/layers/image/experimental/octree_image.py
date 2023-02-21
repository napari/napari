"""OctreeImage class.

An eventual replacement for Image that combines single-scale and
chunked (tiled) multi-scale into one implementation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Set

import numpy as np

from napari.layers.image.experimental._octree_slice import (
    OctreeSlice,
    OctreeView,
)
from napari.layers.image.experimental.octree_chunk import OctreeChunk
from napari.layers.image.experimental.octree_intersection import (
    OctreeIntersection,
)
from napari.layers.image.experimental.octree_level import OctreeLevelInfo
from napari.layers.image.experimental.octree_util import (
    OctreeDisplayOptions,
    OctreeMetadata,
)
from napari.layers.image.image import _ImageBase
from napari.utils.events import Event
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components.experimental.chunk import ChunkRequest

LOGGER = logging.getLogger("napari.octree.image")


class _OctreeImageBase(_ImageBase):
    """Image layer rendered using an octree.

    Experimental variant of Image that renders using an octree. For 2D
    images the octree is really just a quadtree. For 3D volumes it will be
    a real octree. This class is intended to eventually replace the
    existing Image class.

    Notes
    -----
    The original Image class handled single-scale and multi-scale images,
    but they were handled quite differently. And its multi-scale did not
    use chunks or tiles. It worked well on local data, but was basically
    unusable for remote or high latency data.

    OctreeImage always uses chunk/tiles. Today those tiles are always
    "small". However, as a special case, if an image is smaller than the
    max texture size, we could some day allow OctreeImage to set its tile
    size equal to that image size.

    At that point "small" images would be draw with a single texture,
    the same way the old Image class drew then. So it would be very
    efficient.

    But larger images would have multiple chunks/tiles and multiple levels.
    Unlike the original Image class multi-scale, the chunks/tiles mean we
    only have to incrementally load more data as the user pans and zooms.

    The goal is OctreeImage gets renamed to just Image and it efficiently
    handles images of any size. It make take a while to get there.

    Attributes
    ----------
    _view : OctreeView
        Describes a view frustum which implies what portion of the OctreeImage
        needs to be draw.
    _slice : OctreeSlice
        When _set_view_slice() is called we create a OctreeSlice()
        that's looking at some specific slice of the data.
    _display : OctreeDisplayOptions
        Settings for how we draw the octree, such as tile size.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._view: OctreeView = None
        self._slice: OctreeSlice = None
        self._intersection: OctreeIntersection = None
        self._display = OctreeDisplayOptions()

        # super().__init__ will call our _set_view_slice() which is kind
        # of annoying since we are aren't fully constructed yet.
        super().__init__(*args, **kwargs)

        # Call after super().__init__
        self.events.add(octree_level=Event, tile_size=Event)

        # TODO_OCTREE: this is hack that we assign OctreeDisplayOptions
        # this event after super().__init__(). Needs to be cleaned up.
        self._display.loaded_event = self.events.loaded

    def _get_value(self, position):
        """Override Image._get_value(position)."""
        return (0, (0, 0))  # TODO_OCTREE: need to implement this.

    @property
    def loaded(self) -> bool:
        """Has the data for this layer been loaded yet.

        As far as the visual system is concerned we are always "loaded" in
        that we can always be drawn. Because our VispyTiledImageLayer can
        always be drawn. Even if no chunk/tiles are loaded yet.
        """
        return True

    @property
    def _empty(self) -> bool:
        """Is this layer completely empty so it can't be drawn.

        As with self.loaded, we are never really empty. Our VispyTiledImageLayer
        can always be drawn. Even if there is nothing to draw.
        """
        return False

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
    def display(self) -> OctreeDisplayOptions:
        """The display options for this octree image layer."""
        return self._display

    @property
    def tile_size(self) -> int:
        """Return the edge length of single tile, for example 256.

        Returns
        -------
        int
            The edge length of a single tile.
        """
        return self._display.tile_size

    @tile_size.setter
    def tile_size(self, tile_size: int) -> None:
        """Set new tile_size.

        Parameters
        ----------
        tile_size : int
            The new tile size.
        """
        self._display.tile_size = tile_size
        self.events.tile_size()

        self._slice = None  # For now must explicitly delete it
        self.refresh()  # Creates a new slice.

    @property
    def tile_shape(self) -> tuple:
        """Return the shape of a single tile, for example 256x256x3.

        Returns
        -------
        tuple
            The shape of a single tile.
        """
        if self.multiscale:
            init_shape = self.data[0].shape
        else:
            init_shape = self.data.shape

        tile_shape = (self.tile_size, self.tile_size)

        if self.rgb:
            # Add the color dimension (usually 3 or 4)
            tile_shape += (init_shape[-1],)

        return tile_shape

    @property
    def meta(self) -> OctreeMetadata:
        """Information about the current octree.

        Returns
        -------
        OctreeMetadata
            Octree dimensions and other info.
        """
        if self._slice is None:
            return None
        return self._slice.meta

    @property
    def octree_level_info(self) -> OctreeLevelInfo:
        """Information about the current level of the current octree.

        Returns
        -------
        OctreeLevelInfo
            Information about the current octree level.
        """
        if self._slice is None:
            return None
        return self._slice.octree_level_info

    @property
    def data_level(self) -> int:
        """Current level of multiscale.

        The base full resolution image is level 0. The highest and coarsest
        level usually contains only a single tile.
        """
        return self._data_level

    @data_level.setter
    def data_level(self, level: int) -> None:
        """Set the octree level we should be displaying.

        Parameters
        ----------
        level : int
            Display this octree level.
        """
        if self._data_level == level:
            return  # It didn't change.

        # Quickly check for less than 0. We can't check for a level
        # that's too high because the Octree might have extended levels?
        if level < 0:
            raise ValueError(
                trans._(
                    "Octree level {level} is negative.",
                    deferred=True,
                    level=level,
                )
            )

        self._data_level = level
        self.events.octree_level()

        if self._slice is not None:
            # This will raise if the level is too high.
            self._slice.octree_level = level

        self.events.loaded()  # redraw

    @property
    def num_octree_levels(self) -> int:
        """Return the total number of octree levels.

        Returns
        -------
        int
            The number of octree levels.
        """
        return len(self.data)  # Multiscale

    def _new_empty_slice(self) -> None:
        """Initialize the current slice to an empty image.

        Overides Image._new_empty_slice() and does nothing because we don't
        need an empty slice. We create self._slice when
        self._set_view_slice() is called.

        The empty slice was needed to satisfy the old VispyImageLayer that
        used a single ImageVisual. But OctreeImage is drawn with
        VispyTiledImageVisual. It does not need an empty image. It gets
        chunks from our self.drawable_chunks property, and it will just draw
        nothing if that returns an empty list.

        When OctreeImage become the only image class, this can go away.
        """

    def get_drawable_chunks(
        self, drawn_set: Set[OctreeChunk]
    ) -> List[OctreeChunk]:
        """Get the chunks in the current slice which are drawable.

        The visual calls this and then draws what we send it. The call to
        get_intersection() will chose the appropriate level of the octree
        to intersect, and then return all the chunks within the
        intersection with that level.

        These are the "ideal" chunks because they are at the level whose
        resolution best matches the current screen resolution.

        Drawing chunks at a lower level than this will work fine, but it's
        a waste in that those chunks will just be downsampled by the card.
        You won't see any "extra" resolution at all. The card can do this
        super fast, so the issue not such much speed as it is RAM and VRAM.

        In the opposite direction, drawing chunks from a higher, the number
        of chunks and storage goes down quickly. The only issue there is
        visual quality, the imagery might look blurry.

        Parameters
        ----------
        drawn_set : Set[OctreeChunk]
            The chunks that are currently being drawn by the visual.

        Returns
        -------
        List[OctreeChunk]
            The drawable chunks.
        """
        if self._slice is None or self._view is None:
            LOGGER.debug("get_drawable_chunks: No slice or view")
            return []  # There is nothing to draw.

        # TODO_OCTREE: Make this a config option, maybe different
        # expansion_factor each level above the ideal level?
        expansion_factor = 1.1
        view = self._view.expand(expansion_factor)

        # Get the current intersection and save it off.
        self._intersection = self._slice.get_intersection(view)

        if self._intersection is None:
            LOGGER.debug("get_drawable_chunks: Intersection is empty")
            return []  # No chunks to draw.

        # Get the ideal chunks. These are the chunks at the preferred
        # resolution. The ones we ideally want to draw once they are in RAM
        # and in VRAM. When all loading is done, we will draw all the ideal
        # chunks.
        ideal_chunks = self._intersection.get_chunks(create=True)
        ideal_level = self._intersection.level.info.level_index

        # log_chunks("ideal_chunks", ideal_chunks)

        # If we are seting the data level level automatically, then update
        # our level to match what was chosen for the intersection.
        if self._view.auto_level:
            self._data_level = ideal_level

        # The loader will initiate loads on any ideal chunks which are not
        # yet in memory. And it will return the chunks we should draw. The
        # chunks we should draw might be ideal chunks, if they are in
        # memory, but they also might be chunks from higher or lower levels
        # in the octree. In general we try to draw "cover the view" with
        # the "best available" data.
        return self._slice.loader.get_drawable_chunks(
            drawn_set, ideal_chunks, ideal_level
        )

    def _update_draw(
        self, scale_factor, corner_pixels_displayed, shape_threshold
    ) -> None:
        """Override Layer._update_draw completely.

        The base Layer._update_draw does stuff for the legacy multi-scale
        that we don't want. And it calls refresh() which we don't need.

        We create our OctreeView() here which has the corners in it.

        Parameters
        ----------
        scale_factor : float
            Scale factor going from canvas to world coordinates.
        corner_pixels_displayed : array
            Coordinates of the top-left and bottom-right canvas pixels in
            world coordinates.
        shape_threshold : tuple
            Requested shape of field of view in data coordinates.

        """
        # Compute our 2D corners from the incoming n-d corner_pixels
        displayed_sorted = sorted(self._slice_input.displayed)
        data_corners = (
            self._transforms[1:]
            .simplified.set_slice(displayed_sorted)
            .inverse(corner_pixels_displayed)
        )

        # Update our self._view to to capture the state of things right
        # before we are drawn. Our self._view will used by our
        # drawable_chunks() method.
        self._view = OctreeView(data_corners, shape_threshold, self.display)

    def get_intersection(self) -> OctreeIntersection:
        """The the interesection between the current view and the octree.

        Returns
        -------
        OctreeIntersection
            The intersection between the current view and the octree.
        """
        if self._slice is None:
            return None

        return self._slice.get_intersection(self._view)

    def _outside_data_range(self, indices) -> bool:
        """Return True if requested slice is outside of data range.

        Returns
        -------
        bool
            True if requested slice is outside data range.
        """

        extent = self._extent_data
        not_disp = self._slice_input.not_displayed

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

    def _set_view_slice(self) -> None:
        """Set the view given the indices to slice with.

        This replaces Image._set_view_slice() entirely. The hope is eventually
        this class OctreeImage becomes Image. And the non-tiled multiscale
        logic in Image._set_view_slice goes away entirely.
        """
        # Consider non-multiscale data as just having a single level
        from napari.components.experimental.chunk import LayerRef

        multilevel_data = self.data if self.multiscale else [self.data]

        if self._slice is not None:
            # For now bail out so we don't nuke an existing slice which
            # contains an existing octree. Soon we'll need to figure out
            # if we are really changing slices (and need a new octree).
            return

        indices = np.array(self._slice_indices)
        if self._outside_data_range(indices):
            return

        # Indices to get at the data we are currently viewing.
        indices = self._get_slice_indices()

        # TODO_OCTREE: easier way to do this?
        base_shape = multilevel_data[0].shape
        base_shape_2d = [base_shape[i] for i in self._slice_input.displayed]

        layer_ref = LayerRef.from_layer(self)

        meta = OctreeMetadata(
            layer_ref,
            base_shape_2d,
            len(multilevel_data),
            self._display.tile_size,
        )

        # OctreeSlice wants all the levels, but only the dimensions
        # of each level that we are currently viewing.
        slice_data = [level_data[indices] for level_data in multilevel_data]
        layer_ref = LayerRef.from_layer(self)

        # Create the slice, it will create the actual Octree.
        self._slice = OctreeSlice(
            slice_data,
            layer_ref,
            meta,
        )

    def _get_slice_indices(self) -> tuple:
        """Get the slice indices including possible depth for RGB."""
        indices = tuple(self._slice_indices)

        if self.rgb:
            indices += (slice(None),)

        return indices

    def on_chunk_loaded(self, request: ChunkRequest) -> None:
        """An asynchronous ChunkRequest was loaded.

        Override Image.on_chunk_loaded() fully.

        Parameters
        ----------
        request : ChunkRequest
            This request was loaded.
        """
        LOGGER.info(
            "on_chunk_loaded: load=%.3fms elapsed=%.3fms location = %s",
            request.load_ms,
            request.elapsed_ms,
            request.location,
        )

        # Pass it to the slice, it will insert the newly loaded data into
        # the OctreeChunk at the right location.
        if self._slice.on_chunk_loaded(request):
            # Redraw with the new chunk.
            # TODO_OCTREE: Call this at most once per frame? It's a bad
            # idea to call it for every chunk?
            LOGGER.debug("on_chunk_loaded calling loaded()")
            self.events.loaded()

    @property
    def remote_messages(self) -> dict:
        """Messages we should send to remote clients."""
        if self._intersection is None:
            return {}

        return {
            "tile_state": self._intersection.tile_state,
            "tile_config": self._intersection.tile_config,
        }
