"""Tests for components.experimental.chunk."""
import numpy as np
import pytest

from napari.components.experimental.chunk import (
    ChunkLocation,
    ChunkRequest,
    LayerRef,
    chunk_loader,
)
from napari.layers.image import Image


def _create_layer() -> Image:
    """Return a small random Image layer."""
    data = np.random.random((32, 16))
    return Image(data)


def test_base_location():
    """Test the base ChunkLocation class.

    The base ChunkLocation is not really used, only the derived
    ImageLocation and OctreeLocation are, but test it anyway.
    """
    layer_ref1 = LayerRef.from_layer(_create_layer())
    layer_ref2 = LayerRef.from_layer(_create_layer())

    location1a = ChunkLocation(layer_ref1)
    location1b = ChunkLocation(layer_ref1)
    location2 = ChunkLocation(layer_ref2)

    assert location1a == location1b
    assert location1a != location2
    assert location1b != location2


@pytest.mark.async_only
def test_loader():
    """Test ChunkRequest and the ChunkLoader."""

    layer = _create_layer()
    shape = (64, 32)
    transpose_shape = (32, 64)

    # Just load one array.
    data = np.random.random(shape)
    chunks = {'image': data}

    # Give data2 different data.
    data2 = data * 2

    # Create the ChunkRequest.
    location = ChunkLocation.from_layer(layer)
    request = ChunkRequest(location, chunks)

    # Load the ChunkRequest.
    request = chunk_loader.load_request(request)

    # Data should only match data not data2.
    assert np.all(data == request.image.data)
    assert not np.all(data2 == request.image.data)

    # request.image is just short-hand for request.chunks['image']
    assert np.all(request.image.data == request.chunks['image'].data)

    # Since we didn't ask for a thumbnail_source it should just be the image data.
    assert np.all(request.thumbnail_source.data == request.image.data)

    # KeyError for chunks that do not exist.
    with pytest.raises(KeyError):
        request.chunks['missing_chunk_name']

    # Test transpose_chunks()
    request.transpose_chunks((1, 0))
    assert request.image.shape == transpose_shape
