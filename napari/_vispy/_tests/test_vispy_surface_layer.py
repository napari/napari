import numpy as np
import pytest
from vispy.geometry import create_cube

from napari._tests.utils import skip_local_popups
from napari._vispy.layers.surface import VispySurfaceLayer
from napari.components.dims import Dims
from napari.layers import Surface


@pytest.fixture()
def cube_layer():
    vertices, faces, _ = create_cube()
    return Surface((vertices['position'] * 100, faces))


@pytest.mark.parametrize('opacity', [0, 0.3, 0.7, 1])
def test_VispySurfaceLayer(cube_layer, opacity):
    cube_layer.opacity = opacity
    visual = VispySurfaceLayer(cube_layer)
    assert visual.node.opacity == opacity


def test_shading(cube_layer):
    cube_layer._slice_dims(Dims(ndim=3, ndisplay=3))
    cube_layer.shading = 'flat'
    visual = VispySurfaceLayer(cube_layer)
    assert visual.node.shading_filter.attached
    assert visual.node.shading_filter.shading == 'flat'
    cube_layer.shading = 'smooth'
    assert visual.node.shading_filter.shading == 'smooth'


@pytest.mark.parametrize(
    'texture_shape',
    [
        (32, 32),
        (32, 32, 1),
        (32, 32, 3),
        (32, 32, 4),
    ],
    ids=('2D', '1Ch', 'RGB', 'RGBA'),
)
def test_add_texture(cube_layer, texture_shape):
    np.random.seed(0)
    visual = VispySurfaceLayer(cube_layer)
    assert visual._texture_filter is None

    texture = np.random.random(texture_shape).astype(np.float32)
    cube_layer.texture = texture
    # no texture filter initally
    assert visual._texture_filter is None

    # the texture filter is created when texture + texcoords are added
    texcoords = create_cube()[0]['texcoord']
    cube_layer.texcoords = texcoords
    assert visual._texture_filter.attached
    assert visual._texture_filter.enabled

    # texture is flipped for openGL when setting up TextureFilter
    np.testing.assert_allclose(
        visual._texture_filter.texture,
        np.flipud(texture),
    )

    # setting texture or texcoords to None disables the filter
    cube_layer.texture = None
    assert not visual._texture_filter.enabled


def test_change_texture(cube_layer):
    np.random.seed(0)
    visual = VispySurfaceLayer(cube_layer)
    texcoords = create_cube()[0]['texcoord']
    cube_layer.texcoords = texcoords

    texture0 = np.random.random((32, 32, 3)).astype(np.float32)
    cube_layer.texture = texture0
    np.testing.assert_allclose(
        visual._texture_filter.texture,
        np.flipud(texture0),
    )

    texture1 = np.random.random((32, 32, 3)).astype(np.float32)
    cube_layer.texture = texture1
    np.testing.assert_allclose(
        visual._texture_filter.texture,
        np.flipud(texture1),
    )


def test_vertex_colors(cube_layer):
    np.random.seed(0)
    cube_layer._slice_dims(Dims(ndim=3, ndisplay=3))
    visual = VispySurfaceLayer(cube_layer)
    n = len(cube_layer.vertices)

    colors0 = np.random.random((n, 4)).astype(np.float32)
    cube_layer.vertex_colors = colors0
    np.testing.assert_allclose(
        visual.node.mesh_data.get_vertex_colors(),
        colors0,
    )

    colors1 = np.random.random((n, 4)).astype(np.float32)
    cube_layer.vertex_colors = colors1
    np.testing.assert_allclose(
        visual.node.mesh_data.get_vertex_colors(),
        colors1,
    )

    cube_layer.vertex_colors = None
    assert visual.node.mesh_data.get_vertex_colors() is None


@skip_local_popups
def test_check_surface_without_visible_faces(make_napari_viewer):
    points = np.array(
        [
            [0, 0.0, 0.0, 0.0],
            [0, 1.0, 0, 0],
            [0, 1, 1, 0],
            [2, 0.0, 0.0, 0.0],
            [2, 1.0, 0, 0],
            [2, 1, 1, 0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    layer = Surface((points, faces))
    viewer = make_napari_viewer(ndisplay=3)
    viewer.show()
    viewer.add_layer(layer)
    # The following with throw an exception.
    viewer.reset()
