"""Stub ChunkLoader: sync only so far.
"""
from typing import Dict

from ...types import ArrayLike
from ._request import ChunkKey, ChunkRequest


class ChunkLoader:
    """Stub for future ChunkLoader that can do async loads."""

    def create_request(
        self, layer, key: ChunkKey, chunks: Dict[str, ArrayLike]
    ) -> ChunkRequest:
        """Create a ChunkRequest for submission to load_chunk.

        This is a stub and will make more sense in the next version.
        """
        # Return the new request.
        return ChunkRequest(key, chunks)

    def load_chunk(self, request: ChunkRequest) -> ChunkRequest:
        """Stub in the future will do async loads.

        Parameters
        ----------
        request : ChunkRequest
            Contains the array to load from and related info.

        Returns
        -------
        ChunkRequest
            The loaded request.
        """

        request.load_chunks()
        return request


# Global instance
chunk_loader = ChunkLoader()
