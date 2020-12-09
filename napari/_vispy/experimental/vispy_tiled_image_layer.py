"""VispyTiledImageLayer class.

A tiled image that uses TiledImageVisual and TextureAtlas2D so that adding
and removing tiles is faster than if we created a separate ImageVisual for
each tile.
"""
from dataclasses import dataclass
from typing import List

from vispy.scene.visuals import create_visual_node

from ...layers.image.experimental import OctreeChunk
from ...layers.image.experimental.octree_image import OctreeImage
from ...utils.perf import block_timer
from ..vispy_image_layer import VispyImageLayer
from .tile_grid import TileGrid
from .tiled_image_visual import TiledImageVisual

# Create the scene graph Node version of this visual. Visuals are a mix of
# the visual itself and a scene graph node. The scene graph node is what
# can added to the scene and transformed.
TiledImageNode = create_visual_node(TiledImageVisual)


@dataclass
class ChunkStats:
    """Statistics about chunks during the update process."""

    drawable: int = 0
    start: int = 0
    remaining: int = 0
    low: int = 0
    final: int = 0

    @property
    def deleted(self) -> int:
        """How many chunks were deleted."""
        return self.start - self.low

    @property
    def created(self) -> int:
        """How many chunks were created."""
        return self.final - self.low


class VispyTiledImageLayer(VispyImageLayer):
    """A tiled image drawn using a single TiledImageVisual.

    Tiles are rendered using TiledImageVisual which uses a TextureAtlas2D,
    see those classes for more details.

    History
    -------

    An early tiled visual we had created a new ImageVisual for each tile.
    This led to crashes with PyQt5 due to the constant scene graph changes.
    Also each new ImageVisual caused a slow down, the shader build was one
    reason. Finally, rendering was slower because it required a texture
    swap for each tile. This newer VispyTiledImageLayer solves those
    problems.

    Parameters
    ----------
    layer : OctreeImage
        The layer we are drawing.

    Attributes
    ----------
    grid : TileGrid
        Optional grid outlining the tiles.
    """

    def __init__(self, layer: OctreeImage):
        # All tiles are stored in a single TileImageVisual.
        visual = TiledImageNode(tile_shape=layer.tile_shape)

        # Pass our TiledImageVisual to the base class, it will become our
        # self.node which VispyBaseImage holds.
        super().__init__(layer, visual)

        # An optional grid that shows tile borders.
        self.grid = TileGrid(self.node)

        # So we redraw when the layer loads new data.
        self.layer.events.loaded.connect(self._on_loaded)

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles currently being drawn.

        Return
        ------
        int
            The number of tiles currently being drawn.
        """
        return self.node.num_tiles

    def set_data(self, node, data) -> None:
        """Set our image data, not implemented.

        ImageVisual has a set_data() method but we don't. No one can set
        the data for the whole image, that's why it's a tiled image in the
        first place. Instead of set_data() we pull our data one chunk at a
        time by calling self.layer.drawable_chunks in our _update_view()
        method.

        Raises
        ------
        NotImplementedError
            Always raises this.
        """
        raise NotImplementedError()

    def _update_drawn_chunks(self) -> ChunkStats:
        """Add or remove tiles to match the chunks which are currently drawable.

        1) Remove tiles which are no longer drawable.
        2) Create tiles for newly drawable chunks.
        3) Optionally update our grid based on the now drawable chunks.

        Return
        ------
        ChunkStats
            Statistics about the update process.
        """
        # Get what we are currently drawing.
        drawn_chunk_set = self.node.chunk_set

        # Get the currently drawable chunks from the layer. We pass it the
        # drawn_chunk_set because depending on what we are drawing, the layer
        # might want to pass us substitute chunks.
        #
        # For example with quadtree/octree rendering it might send a tile
        # from a higher level. Once the preferred chunk is drawing, it can
        # stop sending us that one.
        drawable_chunks: List[OctreeChunk] = self.layer.get_drawable_chunks(
            drawn_chunk_set
        )

        # Record some stats about this update process, where stats.drawable are
        # the number of drawable chunks we are starting with.
        stats = ChunkStats(drawable=len(drawable_chunks))

        # Create the drawable set of chunks using their keys, so we can
        # check membership quickly.
        # TODO_OCTREE: use __hash__ not OctreeChunk.key? Using __hash__
        # did not immediately work, but we should try again.
        drawable_set = set(
            octree_chunk.key for octree_chunk in drawable_chunks
        )

        # The number of tiles we are currently draw before the update.
        stats.start = self.num_tiles

        # Remove tiles if their chunk is no longer in the drawable set.
        self.node.prune_tiles(drawable_set)

        # The low point, after removing but before adding.
        stats.low = self.num_tiles

        # Remaining is how many tiles in drawable_chunks still need to be
        # added. We don't necessarily add them all in one frame so that we
        # don't tank the framerate.
        stats.remaining = self._add_chunks(drawable_chunks)

        # Final number of tiles we are drawing after adding.
        stats.final = self.num_tiles

        # The grid is only for debugging and demos, yet it's quite useful
        # otherwise you can't really see the borders between the tiles.
        if self.layer.display.show_grid:
            self.grid.update_grid(self.node.octree_chunks)
        else:
            self.grid.clear()

        # if stats.created > 0 or stats.deleted > 0:
        #     for chunk in drawable_chunks:
        #       print(f"drawn: {chunk}")

        return stats

    def _add_chunks(self, drawable_chunks: List[OctreeChunk]) -> int:
        """Add some or all of these drawable chunks to the tiled image.

        Parameters
        ----------
        drawable_chunks : List[OctreeChunk]
            Chunks we should add, if not already in the tiled image.

        Return
        ------
        int
            The number of chunks that still need to be added.
        """
        if not self.layer.display.track_view:
            # Tracking the view is the normal mode, where the tiles load in as
            # the view moves. Not tracking the view is only used for debugging
            # or demos, to show the tiles we "were" showing previously, without
            # updating them.
            return 0  # Nothing more to add

        # Add tiles for drawable chunks that do not already have a tile.
        # This might not add all the chunks, because doing so might
        # tank the framerate.
        #
        # Even though the chunks are already in RAM, we have to do some
        # processing and then we have to move the data to VRAM. That time
        # cost might not happen here, we probably are just queueing up a
        # transfer that will happen when we next call glFlush() to let the
        # card do its business.
        #
        # Any chunks not added this frame will have a chance to be
        # added the next frame, if they are still on the drawable_chunks
        # list next frame. It's important we keep asking the layer for
        # the drawable chunks every frame. We don't want to queue up and
        # add chunks which might no longer be needed, because the
        # camera might move every frame. The situation might be
        # evolving rapidly if the camera is moving a lot.
        return self.node.add_chunks(drawable_chunks)

    def _update_tile_shape(self) -> None:
        """If the tile shape was changed, update our node."""
        # This might be overly dynamic, but for now if we see there's a new
        # tile shape we nuke our texture atlas and start over with the new
        # shape.
        #
        # We added this because the QtTestImage GUI currently set the tile
        # shape after the layer is created. Thus potentially changing the
        # same on the fly. But this "on the fly change" might come in handy
        # and it was not hard to implement, but we could probably drop it
        # if it becomes a pain.
        tile_shape = self.layer.tile_shape
        if self.node.tile_shape != tile_shape:
            self.node.set_tile_shape(tile_shape)

    def _update_view(self) -> int:
        """Update the tiled image based on what's drawable in the layer.

        We call self._update_chunks() which asks the layer what chunks are
        drawable, then it potentially loads some of those chunks, if we
        didn't already have them. This method returns how many drawable
        chunks still need to be added.

        If we return non-zero, we expect to be polled and drawn again, even
        if the camera isn't moving. Polled and drawn until we can finish
        adding the rest of the drawable chunks.

        Return
        ------
        int
             The number of chunks that still need to be added.
        """
        if not self.node.visible:
            return 0

        self._update_tile_shape()  # In case the tile shape changed!

        with block_timer("_update_drawn_chunks") as elapsed:
            stats = self._update_drawn_chunks()

        # Print for now, but switch log file soon. Should be no prints
        # in production code.
        if stats.created > 0 or stats.deleted > 0:
            print(
                f"tiles: {stats.start} -> {stats.final} "
                f"create: {stats.created} delete: {stats.deleted} "
                f"time: {elapsed.duration_ms:.3f}ms"
            )

        return stats.remaining

    def _on_poll(self, event=None) -> None:
        """Called before we are drawn.

        This is called when the camera moves, or when we have chunks that
        need to be loaded. We update which tiles we are drawing based on
        which chunks are currently drawable.
        """
        super()._on_poll()

        # Mark the event "handled" if we have more chunks to load.
        #
        # By saying the poll event was "handled" we're telling QtPoll to
        # keep polling us, even if the camera stops moving. So that we can
        # finish up the loads/draws with a potentially still camera.
        #
        # We'll be polled until no visuals handle the event, meaning
        # no visuals need polling. Then all is quiet until the camera
        # moves again.
        need_polling = self._update_view() > 0
        event.handled = need_polling

    def _on_loaded(self, _event) -> None:
        """The layer loaded new data, so update or view."""
        self._update_view()
