import numpy as np

from napari.layers.image._image_slice import ImageSlice


def _converter(array):
    return array * 2


def test_image_slice():
    """Test ImageSlice and ImageView."""
    image1 = np.random.random((32, 16))
    image2 = np.random.random((32, 16))

    # Create a slice and check it was created as expected.
    image_slice = ImageSlice(image1, _converter)
    assert image_slice.rgb is False
    assert id(image_slice.image.view) == id(image1)
    assert id(image_slice.image.raw) == id(image1)

    # Update the slice and see the conversion happened.
    image_slice.image.raw = image2
    assert id(image_slice.image.raw) == id(image2)
    assert np.all(image_slice.image.view == image2 * 2)

    # Test ImageSlice.set_raw_images().
    image3 = np.random.random((32, 16))
    image4 = np.random.random((32, 16))
    image_slice._set_raw_images(image3, image4)
    assert id(image_slice.image.raw) == id(image3)
    assert id(image_slice.thumbnail.raw) == id(image4)
    assert np.all(image_slice.image.view == image3 * 2)
    assert np.all(image_slice.thumbnail.view == image4 * 2)
