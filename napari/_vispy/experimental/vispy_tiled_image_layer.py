"""VispyTiledImageLayer class.
"""
from collections import namedtuple
from typing import List

from vispy.scene.visuals import create_visual_node

from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..vispy_image_layer import VispyImageLayer
from .tile_grid import TileGrid
from .tiled_image_visual import TiledImageVisual

# Create the scene graph Node version of this visual.
TiledImageNode = create_visual_node(TiledImageVisual)

Stats = namedtuple(
    'Stats', "num_seen num_start num_created num_deleted num_final"
)


class VispyTiledImageLayer(VispyImageLayer):
    """A tiled image drawn using a single TiledImageVisual.

    The original VispyCompoundImageLayer was a tiled image drawn with
    an individual ImageVisual for each tile. That was slow and led
    to crashes with PyQt5 so this version using a single TiledImageVisual
    instead, which uses a TextureAtlas2D.

    That way adding a tile does not result in scene graph changes, a full
    texture upload, or a shader rebuild. So adding tiles is fast, it
    only has to upload the texture data for that one tile.
    """

    def __init__(self, layer):

        # We use our visual instead of ImageVisual's own.
        self.visual = TiledImageNode(tile_shape=layer.tile_shape)
        super().__init__(layer, self.visual)

        # For debugging
        self.grid = TileGrid(self.node)

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles currently being drawn.

        Return
        ------
        int
            The number of tiles currently being drawn.
        """
        return self.visual.num_tiles

    def set_data(self, node, data):
        """Set our data, not implemented."""
        # ImageVisual has a set_data() method but we don't. We pull our
        # data by calling self.layer.visible_chunks in our _update_view()
        # method. And each chunk gets added to our visual as a separate
        # tile.
        raise NotImplementedError()

    def _update_visible_chunks(self) -> None:
        """Add or remove tiles to match the currently visible chunks.

        1) Remove tiles which are no longer visible.
        2) Create tiles for newly visible chunks.
        3) Optionally update our grid to outline visible tiles.
        """
        # Get the currently visible chunk from the layer.
        visible_chunks: List[ChunkData] = self.layer.visible_chunks

        num_seen = len(visible_chunks)

        # Create the visible set of chunks using their keys.
        # TODO_OCTREE: use __hash__ not ChunkData.key?
        visible_set = set(chunk_data.key for chunk_data in visible_chunks)

        num_start = self.num_tiles

        # Remove tiles for chunks which are no longer visible.
        self.visual.prune_tiles(visible_set)

        num_low = self.num_tiles
        num_deleted = num_start - num_low

        if self.layer.track_view:
            # Add tiles for visible chunks that do not already have a tile.
            self.visual.add_chunks(visible_chunks)

        num_final = self.num_tiles
        num_created = num_final - num_low

        if self.layer.show_grid:
            self.grid.update_grid(self.visual.chunk_data)
        else:
            self.grid.clear()

        return Stats(num_seen, num_start, num_created, num_deleted, num_final)

    def _update_tile_shape(self):
        # This might be overly dynamic, but for now if we see there's
        # a new tile shape we nuke our texture atlas and start over
        # with the new tile shape. Maybe there should be some type of
        # more explicit change required?
        tile_shape = self.layer.tile_shape
        if self.visual.tile_shape != tile_shape:
            self.visual.set_tile_shape(tile_shape)

    def _on_camera_move(self, event=None):
        """Called on any camera movement.

        Update tiles based on which chunks are currently visible.
        """
        super()._on_camera_move()

        if not self.node.visible:
            return

        self._update_tile_shape()

        with block_timer("_update_visible_chunks") as elapsed:
            stats = self._update_visible_chunks()

        if stats.num_created > 0 or stats.num_deleted > 0:
            print(
                f"tiles: {stats.num_start} -> {stats.num_final} "
                f"create: {stats.num_created} delete: {stats.num_deleted} "
                f"time: {elapsed.duration_ms:.3f}ms"
            )
