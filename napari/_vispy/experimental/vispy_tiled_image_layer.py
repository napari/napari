"""VispyTiledImageLayer class.
"""
from collections import namedtuple
from typing import List, Set

from ...layers.image.experimental.octree_util import ChunkData
from ...utils.perf import block_timer
from ..vispy_image_layer import VispyImageLayer
from .tile_grid import TileGrid
from .tiled_image_visual import TiledImageVisual

# TODO_OCTREE: hook up to QtRender UI
SHOW_GRID = True

Stats = namedtuple(
    'Stats', "num_seen num_start num_created num_deleted num_final"
)


class ImageTile:
    """Holds the data for one image tile.

    Parameters
    ----------
    chunk : ChunkData
        The data used to create the tile.
    tile_index : int
        the index of this tile in the texture atlas.
    """

    def __init__(self, chunk_data: ChunkData, tile_index: int):
        self.chunk_data = chunk_data
        self.tile_index = tile_index


class VispyTiledImageLayer(VispyImageLayer):
    """A tile image using a single TiledImageVisual."""

    def __init__(self, layer):
        # TODO_OCTREE:
        #
        # Our parent VispyImageLayer creates an ImageVisual that gets
        # passed into VispyBaseLayer and it becomes VispyBaseLayer.node.
        #
        # We're not using this ImageVisual for anything except as a scene
        # graph parent. So we could clean up these 3 classes to get rid
        # of that do-nothing ImageVisual we if we wanted to:
        #
        # VispyTiledImageLayer -> VispyImageLayer -> VispyBaseLayer
        #
        # But it works fine like this.
        super().__init__(layer)

        self.visual = TiledImageVisual(tile_shape=layer.tile_shape)

        if SHOW_GRID:
            self.grid = TileGrid(self.node)

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles in the layer.

        Return
        ------
        int
            The number of tiles in the layer.
        """
        return self.visual.num_tiles

    def _update_view(self) -> None:
        """Add or removes tiles to match the current view.

        The basic algorithm is:
        1) Create ImageTiles for any chunks that are currently visible which
           do not have an ImageTile yet.

        2) Remove ImageTiles for chunks which are no longer visible.

        3) Create the optional grid around only the visible tiles.
        """
        # Get the currently visible chunks.
        visible_chunks = self.layer.visible_chunks

        num_seen = len(visible_chunks)

        # The set is keyed by the chunk's position and level.
        # TODO_OCTREE: use __hash__ not ChunkData.key?
        visible_set = set(c.key for c in visible_chunks)

        num_start = self.num_tiles

        # Remnove tiles for chunks which are no longer visible.
        self._remove_stale_tiles(visible_set)

        num_low = self.num_tiles
        num_deleted = num_start - num_low

        # Add tiles for visible chunks that do not already have a tile.
        self._add_new_tiles(visible_chunks)

        num_final = self.num_tiles
        num_created = num_final - num_low

        if SHOW_GRID:
            self.grid.update_grid(self.visual.chunk_data)

        return Stats(num_seen, num_start, num_created, num_deleted, num_final)

    def _add_new_tiles(self, visible_chunks: List[ChunkData]) -> None:
        """Add tiles for visible chunks that don't already have a tile.

        Parameters
        ----------
        visible_chunks : List[ChunkData]
        """
        if not self.layer.track_view:
            return  # Not actively creating new visuals.

        for chunk_data in visible_chunks:
            if chunk_data not in self.visual:
                self._add_tile(chunk_data)  # Add a tile for this chunk.

    def _add_tile(self, chunk_data: ChunkData) -> ImageTile:
        """Create and return one new ImageTile for this chunk.

        Parameters
        ----------
        chunk : ChunkData
            Create an ImageTile for this chunk.

        Returns
        -------
        ImageTile
            The new ImageTile we created.
        """
        # TODO_OCTREE: we might need to do some (but not all) of the processing
        # in our parent VispyImageLayer._set_node_data() class, but for now
        # we do nothing...
        self.visual.add_tile(chunk_data)

    def _remove_stale_tiles(self, visible_set: Set[ChunkData]) -> None:
        """Remove tiles for chunks which are no longer visible.

        Parameters
        ----------
        visible_set : Set[ChunkData]
            The currently visible chunks.
        """
        self.visual.prune_tiles(visible_set)

    def _on_camera_move(self, event=None):
        super()._on_camera_move()

        with block_timer("_update_view") as elapsed:
            stats = self._update_view()

        if stats.num_created > 0 or stats.num_deleted > 0:
            print(
                f"tiles: {stats.num_start} -> {stats.num_final} "
                f"create: {stats.num_created} delete: {stats.num_deleted} "
                f"total: {elapsed.duration_ms:.3f}ms"
            )
