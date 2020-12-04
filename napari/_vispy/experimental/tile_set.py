"""TileSet class.

TiledImageVisual uses this class to track the tiles its drawing.
"""
from typing import Dict, List, NamedTuple, Set

from ...layers.image.experimental import OctreeChunk, OctreeChunkKey
from .texture_atlas import AtlasTile


class TileData(NamedTuple):
    """TileSet stores one TileState per tile, each one has a TileData.

    Attributes
    ----------
    octree_chunk : OctreeChunk
        The chunk that created the tile.

    atlas_tile : AtlasTile
        The tile that was created from the chunk.
    """

    octree_chunk: OctreeChunk
    atlas_tile: AtlasTile


class TileState:
    """The state stored for every tile in the TileSet.

    Parameters
    ----------
    octree_chunk : OctreeChunk
        The chunk that produced this tile.
    atlas_tile : AtlasTile
        The vert and tex coord information for the tile.

    Attributes
    ----------
    stale : bool
        Stale tiles are going to be replaced soon.
    """

    def __init__(self, octree_chunk: OctreeChunk, atlas_tile: AtlasTile):
        self.data: TileData = TileData(octree_chunk, atlas_tile)
        self.stale: bool = False


class TileSet:
    """The tiles we are drawing.

    Fast test for membership in both directions: dict and a set.

    Attributes
    ----------
    _tiles : Dict[int, TileState]
        Maps tile_index to the the TileStat we have for that tile.
    _chunks : Set[OctreeChunkKey]
        The chunks we have in the set, for fast membership tests.
    """

    def __init__(self):
        self._tiles: Dict[int, TileState] = {}
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

        self._tiles[tile_index] = TileState(octree_chunk, atlas_tile)
        self._chunks.add(octree_chunk.key)

    def mark_stale(self, tile_index: int) -> None:
        """Mark this tile as stale.

        Stale means we might continue to draw the tile, but it's expected
        to be replaced soon. Note with an octree the replacement might be
        physically smaller or bigger. One tile might be replaced by four
        smaller tiles. Or four tiles might be replaced by one bigger tile.

        Parameters
        ----------
        tile_index : int
            The tile to mark stale.
        """

    def remove(self, tile_index: int) -> None:
        """Remove the TileState at this index from the set.

        tile_index : int
            Remove the TileState at this index.
        """
        octree_chunk = self._tiles[tile_index].octree_chunk
        self._chunks.remove(octree_chunk.key)
        del self._tiles[tile_index]

    @property
    def chunks(self) -> List[OctreeChunk]:
        """Return all the chunks we are tracking.

        Return
        ------
        List[OctreeChunk]
            All the chunks in the set.
        """
        return [
            tile_state.data.octree_chunk for tile_state in self._tiles.values()
        ]

    @property
    def tile_state(self) -> List[TileState]:
        """Return the state for all tiles in the set.

        Return
        ------
        List[TileState]
            State for all the tiles in the set.
        """
        return self._tiles.values()

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
