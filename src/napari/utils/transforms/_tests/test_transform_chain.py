import numpy.testing as npt
import pytest

from napari.utils.transforms import (
    Affine,
    CompositeAffine,
    ScaleTranslate,
    TransformChain,
)

transform_types = [Affine, CompositeAffine, ScaleTranslate]


@pytest.mark.parametrize('Transform', transform_types)
def test_transform_chain(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])
    transform_c = transform_b.compose(transform_a)

    transform_chain = TransformChain([transform_a, transform_b])

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_chain(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('Transform', transform_types)
def test_transform_chain_simplified(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain = TransformChain([transform_a, transform_b])
    transform_c = transform_chain.simplified

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_chain(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('Transform', transform_types)
def test_transform_chain_inverse(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain = TransformChain([transform_a, transform_b])
    transform_chain_inverse = transform_chain.inverse

    new_coord = transform_chain(coord)
    orig_coord = transform_chain_inverse(new_coord)
    npt.assert_allclose(coord, orig_coord)


@pytest.mark.parametrize('Transform', transform_types)
def test_transform_chain_slice(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3, 3], translate=[8, 2, -5])
    transform_b = Transform(scale=[0.3, 1, 1.4], translate=[-2.2, 4, 3])
    transform_c = Transform(scale=[2, 3], translate=[8, -5])
    transform_d = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain_a = TransformChain([transform_a, transform_b])
    transform_chain_b = TransformChain([transform_c, transform_d])

    transform_chain_sliced = transform_chain_a.set_slice([0, 2])

    new_coord_1 = transform_chain_sliced(coord)
    new_coord_2 = transform_chain_b(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('Transform', transform_types)
def test_transform_chain_expanded(Transform):
    coord = [10, 3, 13]
    transform_a = Transform(scale=[2, 1, 3], translate=[8, 0, -5])
    transform_b = Transform(scale=[0.3, 1, 1.4], translate=[-2.2, 0, 3])
    transform_c = Transform(scale=[2, 3], translate=[8, -5])
    transform_d = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])

    transform_chain_a = TransformChain([transform_a, transform_b])
    transform_chain_b = TransformChain([transform_c, transform_d])
    transform_chain_expanded = transform_chain_b.expand_dims([1])

    new_coord_2 = transform_chain_a(coord)
    new_coord_1 = transform_chain_expanded(coord)
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_base_transform_init_is_called():
    # TransformChain() was not calling Transform.__init__() at one point.
    # So below would fail with AttributeError: 'TransformChain' object has
    # no attribute 'name'.
    chain = TransformChain()
    assert chain.name is None


def test_setitem_invalidates_cache():
    chain = TransformChain((Affine(scale=(2, 3)), Affine(scale=(4, 5))))

    chain[0] = Affine(scale=(1, -1))

    npt.assert_array_equal(chain((1, 1)), (4, -5))
    npt.assert_array_equal(chain.inverse((1, 1)), (1 / 4, -1 / 5))


def test_delitem_invalidates_cache():
    chain = TransformChain((Affine(scale=(2, 3)), Affine(scale=(4, 5))))

    del chain[1]

    npt.assert_array_equal(chain((1, 1)), (2, 3))
    npt.assert_array_equal(chain.inverse((1, 1)), (1 / 2, 1 / 3))


def test_mutate_item_invalidates_cache():
    chain = TransformChain((Affine(scale=(2, 3)), Affine(scale=(4, 5))))

    chain[0].scale = (1, -1)

    npt.assert_array_equal(chain((1, 1)), (4, -5))
    npt.assert_array_equal(chain.inverse((1, 1)), (1 / 4, -1 / 5))
