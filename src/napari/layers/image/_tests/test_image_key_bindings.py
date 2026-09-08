import numpy as np

from napari.layers.image import Image, _image_key_bindings as key_bindings


def test_auto_contrast_once_sets_clims():
    data = np.array([[0, 10]])
    layer = Image(data)
    # change to something else first
    layer.contrast_limits = (1, 2)
    key_bindings.auto_contrast_once(layer)
    assert layer.contrast_limits == [0.0, 10.0]
