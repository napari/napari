"""VispyTiledImageLayer class.

A tiled image that uses TiledImageVisual and TextureAtlas2D so
adding/removing tiles is extremely fast.
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

    Tiles are rendered using TiledImageVisual which uses a TextureAtlas2D,
    see those classes for more details.

    History
    -------

    An early tiled visual we had created a new ImageVisual for each tile.
    This led to crashes with PyQt5 due to the constant scene graph changes.
    Also each new ImageVisual caused a slow down, the shader build was one
    reason. Finally rendering was slower because it required a texture swap
    for each tile. This new tiled version solves those problems.

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
        time by calling self.layer.visible_chunks in our _update_view()
        method.
        """
        raise NotImplementedError()

    def _update_chunks(self) -> ChunkStats:
        """Add or remove tiles to match the chunks which are currently visible.

        1) Remove tiles which are no longer visible.
        2) Create tiles for newly visible chunks.
        3) Optionally update our grid to outline the visible chunks.
        """
        # Get the currently visible chunks from the layer.
        visible_chunks: List[OctreeChunk] = self.layer.visible_chunks

        stats = ChunkStats(seen=len(visible_chunks))

        # Create the visible set of chunks using their keys.
        # TODO_OCTREE: use __hash__ not OctreeChunk.key?
        visible_set = set(octree_chunk.key for octree_chunk in visible_chunks)

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
            self.grid.update_grid(self.node.octree_chunk)
        else:
            self.grid.clear()

        return stats

    def _update_tile_shape(self) -> None:
        """Check if the tile shape was changed on us."""
        # This might be overly dynamic, but for now if we see there's a new
        # tile shape we nuke our texture atlas and start over with the new
        # tile shape.
        #
        # We added this because the QtTestImage GUI sets the tile shape
        # after the layer is created. But the ability might come in handy
        # and it was not hard to implement.
        tile_shape = self.layer.tile_shape
        if self.node.tile_shape != tile_shape:
            self.node.set_tile_shape(tile_shape)

    def _update_view(self):
        if not self.node.visible:
            return

        self._update_tile_shape()  # In case the tile shape changed!

        with block_timer("_update_chunks") as elapsed:
            stats = self._update_chunks()

        if stats.created > 0 or stats.deleted > 0:
            print(
                f"tiles: {stats.start} -> {stats.final} "
                f"create: {stats.created} delete: {stats.deleted} "
                f"time: {elapsed.duration_ms:.3f}ms"
            )

    def _on_camera_move(self, event=None) -> None:
        """Called on any camera movement.

        Update tiles based on which chunks are currently visible.
        """
        super()._on_camera_move()
        self._update_view()

    def _on_loaded(self, _event):
        self._update_view()
