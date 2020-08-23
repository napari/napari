from napari.layers import Image
from napari._vispy.vispy_image_layer import VispyImageLayer
import numpy as np
from vispy.app import Canvas
from vispy.gloo import gl

c = Canvas(show=False)
GL_MAX_TEXTURE_SIZE = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)


def test_status_bar():
    shapes = [
        (2, 4),
        (12, 3),
        (256, 4048),
        (2, GL_MAX_TEXTURE_SIZE * 2),
        (GL_MAX_TEXTURE_SIZE * 2, 2),
        (2, int(GL_MAX_TEXTURE_SIZE * 1.6)),
        (int(GL_MAX_TEXTURE_SIZE*1.1), int(GL_MAX_TEXTURE_SIZE*1.1))
    ]

    for shape in shapes:
        _test_status_bar(shape)


def _test_status_bar(shape):
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
        layer.coordinates = test_point
        assert layer._get_value() == expected_value


if __name__ == '__main__':
    test_status_bar()
