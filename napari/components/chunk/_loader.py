"""Stub ChunkLoader: sync only so far.
"""
from typing import Dict

from ...layers.base.base import Layer
from ...types import ArrayLike
from ._request import ChunkKey, ChunkRequest


class ChunkLoader:
    """Stub for future ChunkLoader that can do async loads."""

    def create_request(
        self, layer: Layer, key: ChunkKey, chunks: Dict[str, ArrayLike]
    ) -> ChunkRequest:
        """Create a ChunkRequest for submission to load_chunk.

        This is a stub and will make more sense in the next version.

        Parameters
        ----------
        layer : Layer
            We are loading a chunk for this layer.
        key : ChunkKey
            This should identify the chunk uniquely.
        chunks : Dict[str, ArrayLike]
            The arrays we should load.
        """
        # Return the new request.
        return ChunkRequest(key, chunks)

    def load_chunk(self, request: ChunkRequest) -> ChunkRequest:
        """Stub in the future will do async loads.

        Parameters
        ----------
        request : ChunkRequest
            The request that contains the arrays we need to load.

        Returns
        -------
        ChunkRequest
            The request which contains the loaded arrays.
        """

        request.load_chunks()
        return request


# Global instance
chunk_loader = ChunkLoader()
