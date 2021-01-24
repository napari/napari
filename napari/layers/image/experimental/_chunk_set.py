"""ChunkSet class.

Used by the OctreeLoader.
"""
from typing import Dict, List, Set

from .octree_chunk import OctreeChunk, OctreeLocation


class ChunkSet:
    """A set of chunks with fast location membership test.

    We use a dict as an ordered set, and then a set with just the locations
    so OctreeLoader._cancel_futures() can quickly test if a location is
    in the set.
    """

    def __init__(self):
        self._dict: Dict[OctreeChunk, int] = {}
        self._locations: Set[OctreeLocation] = set()

    def __len__(self) -> int:
        """Return the size of the size.

        Returns
        -------
        int
            The size of the set.
        """
        return len(self._dict)

    def __contains__(self, chunk: OctreeChunk) -> bool:
        """Return true if the set contains this chunk.

        Returns
        -------
        bool
            True if the set contains the given chunk.
        """
        return chunk in self._dict

    def add(self, chunks: List[OctreeChunk]) -> None:
        """Add these chunks to the set.

        Parameters
        ----------
        chunks : List[OctreeChunk]
            Add these chunks to the set.
        """
        for chunk in chunks:
            self._dict[chunk] = 1
            self._locations.add(chunk.location)

    def chunks(self) -> List[OctreeChunk]:
        """Get all the chunks in the set.

        Returns
        -------
        List[OctreeChunk]
            All the chunks in the set.
        """
        return self._dict.keys()

    def has_location(self, location: OctreeLocation) -> bool:
        """Return True if the set contains this location.

        Returns
        -------
        bool
            True if the set contains this location.
        """
        return location in self._locations
