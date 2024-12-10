from itertools import permutations

import numpy as np
import numpy.testing as npt
import pytest

from napari._vispy._tests.utils import vispy_image_scene_size
from napari._vispy.layers.image import VispyImageLayer
from napari.components.dims import Dims
from napari.layers import Image


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_2d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 3D.
    """
    image = Image(np.zeros((4, 2)), scale=(1, 2))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((4, 4, 1), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_2d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small square when displayed in 2D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=2, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 0), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2)))
def test_3d_slice_of_3d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((8, 4, 2)), scale=(1, 2, 4))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=3, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((8, 8, 8), scene_size)


@pytest.mark.parametrize('order', permutations((0, 1, 2, 3)))
def test_3d_slice_of_4d_image_with_order(order):
    """See https://github.com/napari/napari/issues/4926

    We define a non-isotropic shape and scale that combined properly
    with any order should make a small cube when displayed in 3D.
    """
    image = Image(np.zeros((16, 8, 4, 2)), scale=(1, 2, 4, 8))
    vispy_image = VispyImageLayer(image)

    image._slice_dims(Dims(ndim=4, ndisplay=3, order=order))

    scene_size = vispy_image_scene_size(vispy_image)
    np.testing.assert_array_equal((16, 16, 16), scene_size)


def test_no_float32_texture_support(monkeypatch):
    """Ensure Image node can be created if OpenGL driver lacks float textures.

    See #3988, #3990, #6652.
    """
    monkeypatch.setattr(
        'napari._vispy.layers.image.get_gl_extensions', lambda: ''
    )
    image = Image(np.zeros((16, 8, 4, 2), dtype='uint8'), scale=(1, 2, 4, 8))
    VispyImageLayer(image)


@pytest.fixture
def im_layer() -> Image:
    return Image(np.zeros((10, 10)))


@pytest.fixture
def pyramid_layer() -> Image:
    return Image([np.zeros((20, 20)), np.zeros((10, 10))])


def test_base_create(im_layer):
    VispyImageLayer(im_layer)


def set_translate(layer):
    layer.translate = (10, 10)


def set_affine_translate(layer):
    layer.affine.translate = (10, 10)
    layer.events.affine()


def set_rotate(layer):
    layer.rotate = 90


def set_affine_rotate(layer):
    layer.affine.rotate = 90
    layer.events.affine()


def no_op(layer):
    pass


@pytest.mark.parametrize(
    ('translate', 'exp_translate'),
    [
        (set_translate, (10, 10)),
        (set_affine_translate, (10, 10)),
        (no_op, (0, 0)),
    ],
    ids=('translate', 'affine_translate', 'no_op'),
)
@pytest.mark.parametrize(
    ('rotate', 'exp_rotate'),
    [
        (set_rotate, ((0, -1), (1, 0))),
        (set_affine_rotate, ((0, -1), (1, 0))),
        (no_op, ((1, 0), (0, 1))),
    ],
    ids=('rotate', 'affine_rotate', 'no_op'),
)
def test_transforming_child_node(
    im_layer, translate, exp_translate, rotate, exp_rotate
):
    layer = VispyImageLayer(im_layer)
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[:2, :2], ((1, 0), (0, 1))
    )
    rotate(im_layer)
    translate(im_layer)
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[:2, :2], ((1, 0), (0, 1))
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (0.5, 0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[:2, :2], exp_rotate
    )
    if translate == set_translate and rotate == set_affine_rotate:
        npt.assert_array_almost_equal(
            layer.node.transform.matrix[-1][:2],
            np.dot(
                np.linalg.inv(exp_rotate),
                np.array([-0.5, -0.5]) + exp_translate,
            ),
        )
    else:
        npt.assert_array_almost_equal(
            layer.node.transform.matrix[-1][:2],
            np.dot(np.linalg.inv(exp_rotate), (-0.5, -0.5)) + exp_translate,
            # np.dot(np.linalg.inv(im_layer.affine.rotate), exp_translate)
        )


def test_transforming_child_node_pyramid(pyramid_layer):
    layer = VispyImageLayer(pyramid_layer)
    corner_pixels_world = np.array([[0, 0], [20, 20]])
    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (0.5, 0.5)
    )
    pyramid_layer.translate = (-10, -10)
    pyramid_layer._update_draw(
        scale_factor=1,
        corner_pixels_displayed=corner_pixels_world,
        shape_threshold=(10, 10),
    )

    npt.assert_array_almost_equal(
        layer.node.transform.matrix[-1][:2], (-0.5, -0.5)
    )
    npt.assert_array_almost_equal(
        layer.node.children[0].transform.matrix[-1][:2], (-9.5, -9.5)
    )


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('ndim', [3, 4])
@pytest.mark.parametrize('ndisplay', [2, 3])
def test_node_origin_is_consistent_with_multiscale(
    scale: int, ndim: int, ndisplay: int
):
    """See https://github.com/napari/napari/issues/6320"""
    scales = (scale,) * ndim

    # Define multi-scale image data with two levels where the
    # higher resolution is twice as high as the lower resolution.
    image = Image(
        data=[np.zeros((8,) * ndim), np.zeros((4,) * ndim)], scale=scales
    )
    vispy_image = VispyImageLayer(image)

    # Take a full slice at the highest resolution.
    image.corner_pixels = np.array([[0] * ndim, [8] * ndim])
    image._data_level = 0
    # Use a slice point of (1, 0, 0, ...) to have some non-zero slice coordinates.
    point = (1,) + (0,) * (ndim - 1)
    image._slice_dims(Dims(ndim=ndim, ndisplay=ndisplay, point=point))
    # Map the node's data origin to a vispy scene coordinate.
    high_res_origin = vispy_image.node.transform.map((0,) * ndisplay)

    # Take a full slice at the lowest resolution and map the origin again.
    image.corner_pixels = np.array([[0] * ndim, [4] * ndim])
    image._data_level = 1
    image._slice_dims(Dims(ndim=ndim, ndisplay=ndisplay, point=point))
    low_res_origin = vispy_image.node.transform.map((0,) * ndisplay)

    # The exact origin may depend on certain parameter values, but the
    # full high and low resolution slices should always map to the same
    # scene origin, since this defines the start of the visible extent.
    np.testing.assert_array_equal(high_res_origin, low_res_origin)
