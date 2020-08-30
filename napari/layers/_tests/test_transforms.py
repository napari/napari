import numpy.testing as npt
import pytest

from napari.layers.transforms import Affine, ScaleTranslate


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate(Transform):
    coord = [10, 13]
    transform = Transform(scale=[2, 3], translate=[8, -5], name='st')
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    assert transform.name == 'st'
    npt.assert_allclose(new_coord, target_coord)


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate_inverse(Transform):
    coord = [10, 13]
    transform = Transform(scale=[2, 3], translate=[8, -5])
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    npt.assert_allclose(new_coord, target_coord)

    inverted_new_coord = transform.inverse(new_coord)
    npt.assert_allclose(inverted_new_coord, coord)


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate_compose(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate_slice(Transform):
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[2, 1, 3], translate=[8, 3, -5], name='st')
    npt.assert_allclose(transform_b.set_slice([0, 2]).scale, transform_a.scale)
    npt.assert_allclose(
        transform_b.set_slice([0, 2]).translate, transform_a.translate
    )
    assert transform_b.set_slice([0, 2]).name == 'st'


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate_expand_dims(Transform):
    transform_a = Transform(scale=[2, 3], translate=[8, -5], name='st')
    transform_b = Transform(scale=[2, 1, 3], translate=[8, 0, -5])
    npt.assert_allclose(transform_a.expand_dims([1]).scale, transform_b.scale)
    npt.assert_allclose(
        transform_a.expand_dims([1]).translate, transform_b.translate
    )
    assert transform_a.expand_dims([1]).name == 'st'


@pytest.mark.parametrize('Transform', [ScaleTranslate, Affine])
def test_scale_translate_identity_default(Transform):
    coord = [10, 13]
    transform = Transform()
    new_coord = transform(coord)
    npt.assert_allclose(new_coord, coord)


def test_scale_translate_rotate():
    coord = [10, 13]
    transform = Affine(
        scale=[2, 3], translate=[8, -5], rotate=90, degrees=True
    )
    new_coord = transform(coord)
    pre_rotate = [2 * 10, 3 * 13]
    # As rotate by 90 degrees, can use [y, -x]
    post_rotate = [pre_rotate[1], -pre_rotate[0]]
    target_coord = [post_rotate[0] + 8, post_rotate[1] - 5]
    npt.assert_allclose(new_coord, target_coord)


def test_affine_properties():
    transform = Affine(
        scale=[2, 3], translate=[8, -5], rotate=90, shear=[1], degrees=True
    )
    npt.assert_allclose(transform.translate, [8, -5])
    npt.assert_allclose(transform.scale, [2, 3])
    npt.assert_almost_equal(transform.rotate, [[0, -1], [1, 0]])
    npt.assert_almost_equal(transform.shear, [1])


def test_scale_translate_rotate_inverse():
    coord = [10, 13]
    transform = Affine(
        scale=[2, 3], translate=[8, -5], rotate=90, degrees=True
    )
    new_coord = transform(coord)
    pre_rotate = [2 * 10, 3 * 13]
    # As rotate by 90 degrees, can use [y, -x]
    post_rotate = [pre_rotate[1], -pre_rotate[0]]
    target_coord = [post_rotate[0] + 8, post_rotate[1] - 5]
    npt.assert_allclose(new_coord, target_coord)

    inverted_new_coord = transform.inverse(new_coord)
    npt.assert_allclose(inverted_new_coord, coord)


def test_scale_translate_rotate_compose():
    coord = [10, 13]
    transform_a = Affine(
        scale=[2, 3], translate=[8, -5], rotate=25, degrees=True
    )
    transform_b = Affine(
        scale=[0.3, 1.4], translate=[-2.2, 3], rotate=65, degrees=True
    )
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)
