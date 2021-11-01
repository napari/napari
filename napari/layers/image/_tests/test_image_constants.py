from napari.layers.image._image_constants import Rendering


def test_rendering_image():
    """Image layer does not use the iso categorical rendering mode"""
    image_subset = Rendering.image_layer_subset()

    assert Rendering.ISO_CATEGORICAL not in image_subset


def test_rendering_labels():
    """Labels layer should just use transluscent and iso_categorical rendering"""
    labels_subset = Rendering.labels_layer_subset()

    assert Rendering.TRANSLUCENT in labels_subset
    assert Rendering.ISO_CATEGORICAL in labels_subset
