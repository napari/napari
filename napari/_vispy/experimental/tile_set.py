"""TileSet class.
"""
from dataclasses import dataclass
from typing import List

from ...layers.image.experimental.octree_util import ChunkData
from .texture_atlas import AtlasTile


@dataclass
class TileData:
    """Statistics about chunks during the update process."""

    chunk_data: ChunkData  # The data that produced this tile.
    atlas_tile: AtlasTile  # Information from the texture atlas.


class TileSet:
    """The tiles we are drawing.

    Fast test for membership in both directions: dict and a set.
    """

    def __init__(self):
        self._tiles = {}
        self._chunks = set()

    def __len__(self) -> int:
        """Return the number of tiles in the set.

        Return
        ------
        int
            The number of tiles in the set.
        """
        return len(self._tiles)

    def clear(self) -> None:
        """Clear out all our tiles and chunks. Forget everything."""
        self._tiles.clear()
        self._chunks.clear()

    def add(self, chunk_data: ChunkData, atlas_tile: AtlasTile) -> None:
        """Add this TiledData to the set.

        Parameters
        ----------
        tile_data : TileData
            Add this to the set.
        """
        tile_index = atlas_tile.index
        self._tiles[tile_index] = TileData(chunk_data, atlas_tile)
        self._chunks.add(chunk_data.key)

    def remove(self, tile_index: int) -> None:
        """Remove the TileData at this index from the set.

        tile_index : int
            Remove the TileData at this index.
        """
        chunk_data = self._tiles[tile_index].chunk_data
        del self._tiles[tile_index]
        self._chunks.remove(chunk_data.key)

    @property
    def chunks(self) -> List[ChunkData]:
        """Return all the chunk data that we have.

        Return
        ------
        List[ChunkData]
            All the chunk data in the set.
        """
        return [tile_data.chunk_data for tile_data in self._tiles.values()]

    @property
    def tile_data(self) -> List[TileData]:
        """Return all the tile data in the set.

        Return
        ------
        List[TileData]
            All the tile data in the set.
        """
        return self._tiles.values()

    def contains_chunk_data(self, chunk_data: ChunkData) -> bool:
        """Return True if the set contains this chunk data.

        Parameters
        ----------
        chunk_data : ChunkData
            Check if ChunkData is in the set.

        Return
        ------
        bool
            True if the set contains this chunk data.
        """
        return chunk_data.key in self._chunks
