"""TileSet class.

TiledImageVisual uses this class to track the tiles it's drawing.
"""
from typing import Dict, List, NamedTuple, Set

from ...layers.image.experimental import OctreeChunk, OctreeChunkKey
from .texture_atlas import AtlasTile


class TileData(NamedTuple):
    """TileSet stores one TileData per tile.

    Attributes
    ----------
    octree_chunk : OctreeChunk
        The chunk that created the tile.

    atlas_tile : AtlasTile
        The tile that was created from the chunk.
    """

    octree_chunk: OctreeChunk
    atlas_tile: AtlasTile


class TileSet:
    """The tiles we are drawing.

    Fast test for membership in both directions: dict and a set.

    Attributes
    ----------
    _tiles : Dict[int, TileData]
        Maps tile_index to the the TileData we have for that tile.
    _chunks : Set[OctreeChunkKey]
        The chunks we have in the set, for fast membership tests.
    """

    def __init__(self):
        self._tiles: Dict[int, TileData] = {}
        self._chunks: Set[OctreeChunkKey] = set()

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

    def add(self, octree_chunk: OctreeChunk, atlas_tile: AtlasTile) -> None:
        """Add this TiledData to the set.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            The chunk we are adding to the tile set.
        atlas_tile : AtlasTile
            The atlas tile that was created for this chunks.
        """
        tile_index = atlas_tile.index

        self._tiles[tile_index] = TileData(octree_chunk, atlas_tile)
        self._chunks.add(octree_chunk.key)

    def remove(self, tile_index: int) -> None:
        """Remove the TileData at this index from the set.

        tile_index : int
            Remove the TileData at this index.
        """
        octree_chunk = self._tiles[tile_index].octree_chunk
        self._chunks.remove(octree_chunk.key)
        del self._tiles[tile_index]

    @property
    def chunk_set(self) -> Set[OctreeChunkKey]:
        """Return the set of chunks we drawing.

        Return
        ------
        Set[OctreeChunkKey]
            The set of chunks we are drawing.
        """
        return self._chunks

    @property
    def chunks(self) -> List[OctreeChunk]:
        """Return all the chunks we are tracking.

        Return
        ------
        List[OctreeChunk]
            All the chunks in the set.
        """
        return [tile_data.octree_chunk for tile_data in self._tiles.values()]

    @property
    def tile_data(self) -> List[TileData]:
        """Return the data for all tiles in the set, unsorted.

        Return
        ------
        List[TileData]
            Data for all the tiles in the set sorted back to front.
        """
        return self._tiles.values()

    @property
    def tile_data_sorted(self) -> List[TileData]:
        """Return the data for all tiles in the set, sorted back to front.

        We return tiles from higher octree levels first. These are the
        larger coarser tiles. These are "the background" while smaller
        higher resolution tiles are drawn in front. So we show the "best
        available" data in all locations.

        Return
        ------
        List[TileData]
            Data for all the tiles in the set sorted back to front.
        """
        return sorted(
            self._tiles.values(),
            key=lambda x: x.octree_chunk.location.level_index,
            reverse=True,
        )

    def contains_octree_chunk(self, octree_chunk: OctreeChunk) -> bool:
        """Return True if the set contains this chunk.

        Parameters
        ----------
        octree_chunk : OctreeChunk
            Check if this chunk is in the set.

        Return
        ------
        bool
            True if the set contains this chunk data.
        """
        return octree_chunk.key in self._chunks
