from napari.layers import Image
from napari._vispy.vispy_image_layer import VispyImageLayer
import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl
import pytest

c = Canvas(show=False)
GL_MAX_TEXTURE_SIZE = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)


@pytest.mark.parametrize("shape", [(2, 4),
                                   (12, 3),
                                   (256, 4048),
                                   (2, GL_MAX_TEXTURE_SIZE * 2),
                                   (GL_MAX_TEXTURE_SIZE * 2, 2),
                                   (2, int(GL_MAX_TEXTURE_SIZE * 1.6)),
                                   # (int(GL_MAX_TEXTURE_SIZE * 1.1), int(GL_MAX_TEXTURE_SIZE * 1.1)),
                                   ])
@pytest.mark.filterwarnings("ignore:data shape:UserWarning")
def test_status_bar(shape):
    data = np.zeros(shape)
    data[shape[0] // 2:, shape[1] // 2:] = 1
    layer = Image(data)
    vispy_layer = VispyImageLayer(layer)

    coordinates_shape = layer._transforms.inverse.simplified(data.shape)

    test_points = [(int(coordinates_shape[0] * 0.25), int(coordinates_shape[1] * 0.25)),
                   (int(coordinates_shape[0] * 0.75), int(coordinates_shape[1] * 0.25)),
                   (int(coordinates_shape[0] * 0.25), int(coordinates_shape[1] * 0.75)),
                   (int(coordinates_shape[0] * 0.75), int(coordinates_shape[1] * 0.75)),
                   ]
    expected_values = [0.0, 0.0, 0.0, 1.0]

    for test_point, expected_value in zip(test_points, expected_values):
        layer.position = test_point
        assert layer.get_value() == expected_value
