import numpy as np

from napari.layers.image import Image


def _create_layer() -> Image:
    """Return a small random Image layer."""
    data = np.random.random((32, 16))
    return Image(data)


def test_image_location():
    """Test the pre-octree ImageLocation class.

    An ImageLocation is just BaseLocation plus indices.
    """
    from napari.layers.image.experimental._image_location import ImageLocation

    layer1 = _create_layer()
    layer2 = _create_layer()

    locations1_0 = (
        ImageLocation(layer1, (0, 0)),
        ImageLocation(layer1, (0, 0)),
    )

    locations1_1 = (
        ImageLocation(layer1, (0, 1)),
        ImageLocation(layer1, (0, 1)),
    )

    locations2_0 = (
        ImageLocation(layer2, (0, 0)),
        ImageLocation(layer2, (0, 0)),
    )

    locations2_1 = (
        ImageLocation(layer2, (0, 1)),
        ImageLocation(layer2, (0, 1)),
    )

    # All identical pairs should be the same.
    assert locations1_0[0] == locations1_0[1]
    assert locations1_1[0] == locations1_1[1]
    assert locations2_0[0] == locations2_0[1]
    assert locations2_1[0] == locations2_1[1]

    # Nothing else should be the same
    for i in range(0, 2):
        assert locations1_0[i] != locations1_1[i]
        assert locations1_0[i] != locations2_0[i]
        assert locations1_0[i] != locations2_1[i]
