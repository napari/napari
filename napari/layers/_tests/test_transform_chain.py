import numpy.testing as npt

from napari.layers.transforms import ScaleTranslate, TransformChain


def test_transform_chain():
    coord = [10, 13]
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])
    transform_c = transform_b.compose(transform_a)

    transform_chain = TransformChain([transform_a, transform_b])

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_chain(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_transform_chain_simplified():
    coord = [10, 13]
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain = TransformChain([transform_a, transform_b])
    transform_c = transform_chain.simplified

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_chain(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_transform_chain_inverse():
    coord = [10, 13]
    transform_a = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain = TransformChain([transform_a, transform_b])
    transform_chain_inverse = transform_chain.inverse

    new_coord = transform_chain(coord)
    orig_coord = transform_chain_inverse(new_coord)
    npt.assert_allclose(coord, orig_coord)


def test_transform_chain_slice():
    coord = [10, 13]
    transform_a = ScaleTranslate(scale=[2, 3, 3], translate=[8, 2, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1, 1.4], translate=[-2.2, 4, 3])
    transform_c = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_d = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain_a = TransformChain([transform_a, transform_b])
    transform_chain_b = TransformChain([transform_c, transform_d])

    transform_chain_sliced = transform_chain_a.set_slice([0, 2])

    new_coord_1 = transform_chain_sliced(coord)
    new_coord_2 = transform_chain_b(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_transform_chain_expanded():
    coord = [10, 3, 13]
    transform_a = ScaleTranslate(scale=[2, 1, 3], translate=[8, 0, -5])
    transform_b = ScaleTranslate(scale=[0.3, 1, 1.4], translate=[-2.2, 0, 3])
    transform_c = ScaleTranslate(scale=[2, 3], translate=[8, -5])
    transform_d = ScaleTranslate(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain_a = TransformChain([transform_a, transform_b])
    transform_chain_b = TransformChain([transform_c, transform_d])
    transform_chain_expandded = transform_chain_b.expand_dims([1])

    new_coord_2 = transform_chain_a(coord)
    new_coord_1 = transform_chain_expandded(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)
