"""VispyTiledImageLayer class.

A tiled image that uses TiledImageVisual and TextureAtlas2D so
adding/removing tiles is extremely fast.
"""
from dataclasses import dataclass
from typing import List

from vispy.scene.visuals import create_visual_node

from ...layers.image.experimental.octree_image import OctreeImage
from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..vispy_image_layer import VispyImageLayer
from .tile_grid import TileGrid
from .tiled_image_visual import TiledImageVisual

# Create the scene graph Node version of this visual.
TiledImageNode = create_visual_node(TiledImageVisual)


@dataclass
class ChunkStats:
    """Statistics about chunks during the update process."""

    seen: int = 0
    start: int = 0
    deleted: int = 0
    low: int = 0
    created: int = 0
    final: int = 0


class VispyTiledImageLayer(VispyImageLayer):
    """A tiled image drawn using a single TiledImageVisual.

    The original VispyCompoundImageLayer was a tiled image drawn with an
    individual ImageVisual for each tile. That was slow and led to crashes
    with PyQt5. This version uses a single TiledImageVisual instead, which
    stores the texture tiles in a TextureAtlas2D.

    The benefit is adding or removing tiles does not cause any scene graph
    changes. And it does not cause the shader to be rebuilt either. It's
    efficient because we only send one tile's worth of data to the card at
    a time. Only the tile's footprint in the atlas texture gets modified.

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
        # self.node which is VispyBaseImage.node.
        super().__init__(layer, visual)

        # Optional grid shows tile borders.
        self.grid = TileGrid(self.node)

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles currently being drawn.

        Return
        ------
        int
            The number of tiles currently being drawn.
        """
        return self.node.num_tiles

    def set_data(self, node, data):
        """Set our data, not implemented."""
        # ImageVisual has a set_data() method but we don't. No one can set
        # the data for the whole image! We pull our data one chunk at a
        # time by calling self.layer.visible_chunks in our _update_view()
        # method.
        raise NotImplementedError()

    def _update_visible_chunks(self) -> ChunkStats:
        """Add or remove tiles to match the chunks which are currently visible.

        1) Remove tiles which are no longer visible.
        2) Create tiles for newly visible chunks.
        3) Optionally update our grid to outline the visible chunks.
        """
        # Get the currently visible chunks from the layer.
        visible_chunks: List[ChunkData] = self.layer.visible_chunks

        stats = ChunkStats(seen=len(visible_chunks))

        # Create the visible set of chunks using their keys.
        # TODO_OCTREE: use __hash__ not ChunkData.key?
        visible_set = set(chunk_data.key for chunk_data in visible_chunks)

        stats.start = self.num_tiles

        # Remove tiles for chunks which are no longer visible.
        self.node.prune_tiles(visible_set)

        stats.low = self.num_tiles
        stats.deleted = stats.start - stats.low

        if self.layer.track_view:
            # Add tiles for visible chunks that do not already have a tile.
            self.node.add_chunks(visible_chunks)

        stats.final = self.num_tiles
        stats.created = stats.final - stats.low

        if self.layer.show_grid:
            self.grid.update_grid(self.node.chunk_data)
        else:
            self.grid.clear()

        return stats

    def _update_tile_shape(self):
        # This might be overly dynamic, but for now if we see there's
        # a new tile shape we nuke our texture atlas and start over
        # with the new tile shape. Maybe there should be some type of
        # more explicit change required?
        tile_shape = self.layer.tile_shape
        if self.node.tile_shape != tile_shape:
            self.node.set_tile_shape(tile_shape)

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

        if stats.created > 0 or stats.deleted > 0:
            print(
                f"tiles: {stats.start} -> {stats.final} "
                f"create: {stats.created} delete: {stats.deleted} "
                f"time: {elapsed.duration_ms:.3f}ms"
            )
