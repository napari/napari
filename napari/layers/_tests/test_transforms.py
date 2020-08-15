import numpy.testing as npt

from napari.layers.transforms import ScaleTranslate


def test_scale_translate():
    coord = [10, 13]
    transform = ScaleTranslate(scale=[2, 3], translate=[8, -5], name='st')
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    assert transform.name == 'st'
    npt.assert_allclose(new_coord, target_coord)


def test_scale_translate_inverse():
    coord = [10, 13]
    transform = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    npt.assert_allclose(new_coord, target_coord)

    inverted_new_coord = transform.inverse(new_coord)
    npt.assert_allclose(inverted_new_coord, coord)


def test_scale_translate_compose():
    coord = [10, 13]
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_scale_translate_slice():
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_b = ScaleTranslate(
        scale=[2, 1, 3], translate=[8, 3, -5], name='st'
    )
    npt.assert_allclose(transform_b.set_slice([0, 2]).scale, transform_a.scale)
    npt.assert_allclose(
        transform_b.set_slice([0, 2]).translate, transform_a.translate
    )
    assert transform_b.set_slice([0, 2]).name == 'st'


def test_scale_translate_expand_dims():
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5], name='st')
    transform_b = ScaleTranslate(scale=[2, 1, 3], translate=[8, 0, -5])
    npt.assert_allclose(transform_a.expand_dims([1]).scale, transform_b.scale)
    npt.assert_allclose(
        transform_a.expand_dims([1]).translate, transform_b.translate
    )
    assert transform_a.expand_dims([1]).name == 'st'


def test_scale_translate_identity_default():
    coord = [10, 13]
    transform = ScaleTranslate()
    new_coord = transform(coord)
    npt.assert_allclose(new_coord, coord)
