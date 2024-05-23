import numpy as np
import numpy.testing as npt
import pytest

from napari._vispy.layers.image import VispyImageLayer
from napari.layers import Image


@pytest.fixture()
def im_layer():
    return Image(np.zeros((10, 10)))


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
