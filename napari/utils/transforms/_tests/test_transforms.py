import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import special_ortho_group

from napari.utils.transforms import Affine, CompositeAffine, ScaleTranslate

transform_types = [Affine, CompositeAffine, ScaleTranslate]


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate(Transform):
    coord = [10, 13]
    transform = Transform(scale=[2, 3], translate=[8, -5], name='st')
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    assert transform.name == 'st'
    npt.assert_allclose(new_coord, target_coord)


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_broadcast_scale(Transform):
    coord = [1, 10, 13]
    transform = Transform(scale=[4, 2, 3], translate=[8, -5], name='st')
    new_coord = transform(coord)
    target_coord = [4, 2 * 10 + 8, 3 * 13 - 5]
    assert transform.name == 'st'
    npt.assert_allclose(transform.scale, [4, 2, 3])
    npt.assert_allclose(transform.translate, [0, 8, -5])
    npt.assert_allclose(new_coord, target_coord)


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_broadcast_translate(Transform):
    coord = [1, 10, 13]
    transform = Transform(scale=[2, 3], translate=[5, 8, -5], name='st')
    new_coord = transform(coord)
    target_coord = [6, 2 * 10 + 8, 3 * 13 - 5]
    assert transform.name == 'st'
    npt.assert_allclose(transform.scale, [1, 2, 3])
    npt.assert_allclose(transform.translate, [5, 8, -5])
    npt.assert_allclose(new_coord, target_coord)


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_inverse(Transform):
    coord = [10, 13]
    transform = Transform(scale=[2, 3], translate=[8, -5])
    new_coord = transform(coord)
    target_coord = [2 * 10 + 8, 3 * 13 - 5]
    npt.assert_allclose(new_coord, target_coord)

    inverted_new_coord = transform.inverse(new_coord)
    npt.assert_allclose(inverted_new_coord, coord)


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_compose(Transform):
    coord = [10, 13]
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[0.3, 1.4], translate=[-2.2, 3])
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_slice(Transform):
    transform_a = Transform(scale=[2, 3], translate=[8, -5])
    transform_b = Transform(scale=[2, 1, 3], translate=[8, 3, -5], name='st')
    npt.assert_allclose(transform_b.set_slice([0, 2]).scale, transform_a.scale)
    npt.assert_allclose(
        transform_b.set_slice([0, 2]).translate, transform_a.translate
    )
    assert transform_b.set_slice([0, 2]).name == 'st'


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_expand_dims(Transform):
    transform_a = Transform(scale=[2, 3], translate=[8, -5], name='st')
    transform_b = Transform(scale=[2, 1, 3], translate=[8, 0, -5])
    npt.assert_allclose(transform_a.expand_dims([1]).scale, transform_b.scale)
    npt.assert_allclose(
        transform_a.expand_dims([1]).translate, transform_b.translate
    )
    assert transform_a.expand_dims([1]).name == 'st'


@pytest.mark.parametrize('Transform', transform_types)
def test_scale_translate_identity_default(Transform):
    coord = [10, 13]
    transform = Transform()
    new_coord = transform(coord)
    npt.assert_allclose(new_coord, coord)


def test_affine_properties():
    transform = Affine(scale=[2, 3], translate=[8, -5], rotate=90, shear=[1])
    npt.assert_allclose(transform.translate, [8, -5])
    npt.assert_allclose(transform.scale, [2, 3])
    npt.assert_almost_equal(transform.rotate, [[0, -1], [1, 0]])
    npt.assert_almost_equal(transform.shear, [1])


def test_affine_properties_setters():
    transform = Affine()
    transform.translate = [8, -5]
    npt.assert_allclose(transform.translate, [8, -5])
    transform.scale = [2, 3]
    npt.assert_allclose(transform.scale, [2, 3])
    transform.rotate = 90
    npt.assert_almost_equal(transform.rotate, [[0, -1], [1, 0]])
    transform.shear = [1]
    npt.assert_almost_equal(transform.shear, [1])


def test_rotate():
    coord = [10, 13]
    transform = Affine(rotate=90)
    new_coord = transform(coord)
    # As rotate by 90 degrees, can use [-y, x]
    target_coord = [-coord[1], coord[0]]
    npt.assert_allclose(new_coord, target_coord)


def test_scale_translate_rotate():
    coord = [10, 13]
    transform = Affine(scale=[2, 3], translate=[8, -5], rotate=90)
    new_coord = transform(coord)
    post_scale = np.multiply(coord, [2, 3])
    # As rotate by 90 degrees, can use [-y, x]
    post_rotate = [-post_scale[1], post_scale[0]]
    target_coord = np.add(post_rotate, [8, -5])
    npt.assert_allclose(new_coord, target_coord)


def test_scale_translate_rotate_inverse():
    coord = [10, 13]
    transform = Affine(scale=[2, 3], translate=[8, -5], rotate=90)
    new_coord = transform(coord)
    post_scale = np.multiply(coord, [2, 3])
    # As rotate by 90 degrees, can use [-y, x]
    post_rotate = [-post_scale[1], post_scale[0]]
    target_coord = np.add(post_rotate, [8, -5])
    npt.assert_allclose(new_coord, target_coord)

    inverted_new_coord = transform.inverse(new_coord)
    npt.assert_allclose(inverted_new_coord, coord)


def test_scale_translate_rotate_compose():
    coord = [10, 13]
    transform_a = Affine(scale=[2, 3], translate=[8, -5], rotate=25)
    transform_b = Affine(scale=[0.3, 1.4], translate=[-2.2, 3], rotate=65)
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)


def test_scale_translate_rotate_shear_compose():
    coord = [10, 13]
    transform_a = Affine(scale=[2, 3], translate=[8, -5], rotate=25, shear=[1])
    transform_b = Affine(
        scale=[0.3, 1.4],
        translate=[-2.2, 3],
        rotate=65,
        shear=[-0.5],
    )
    transform_c = transform_b.compose(transform_a)

    new_coord_1 = transform_c(coord)
    new_coord_2 = transform_b(transform_a(coord))
    npt.assert_allclose(new_coord_1, new_coord_2)


@pytest.mark.parametrize('dimensionality', [2, 3])
def test_affine_matrix(dimensionality):
    np.random.seed(0)
    N = dimensionality
    A = np.eye(N + 1)
    A[:-1, :-1] = np.random.random((N, N))
    A[:-1, -1] = np.random.random(N)

    # Create transform
    transform = Affine(affine_matrix=A)

    # Check affine was passed correctly
    np.testing.assert_almost_equal(transform.affine_matrix, A)

    # Create input vector
    x = np.ones(N + 1)
    x[:-1] = np.random.random(N)

    # Apply transform and direct matrix multiplication
    result_transform = transform(x[:-1])
    result_mat_multiply = (A @ x)[:-1]

    np.testing.assert_almost_equal(result_transform, result_mat_multiply)


@pytest.mark.parametrize('dimensionality', [2, 3])
def test_affine_matrix_compose(dimensionality):
    np.random.seed(0)
    N = dimensionality
    A = np.eye(N + 1)
    A[:-1, :-1] = np.random.random((N, N))
    A[:-1, -1] = np.random.random(N)

    B = np.eye(N + 1)
    B[:-1, :-1] = np.random.random((N, N))
    B[:-1, -1] = np.random.random(N)

    # Create transform
    transform_A = Affine(affine_matrix=A)
    transform_B = Affine(affine_matrix=B)

    # Check affine was passed correctly
    np.testing.assert_almost_equal(transform_A.affine_matrix, A)
    np.testing.assert_almost_equal(transform_B.affine_matrix, B)

    # Compose tranform and directly matrix multiply
    transform_C = transform_B.compose(transform_A)
    C = B @ A
    np.testing.assert_almost_equal(transform_C.affine_matrix, C)


@pytest.mark.parametrize('dimensionality', [2, 3])
def test_numpy_array_protocol(dimensionality):
    N = dimensionality
    A = np.eye(N + 1)
    A[:-1] = np.random.random((N, N + 1))
    transform = Affine(affine_matrix=A)
    np.testing.assert_almost_equal(transform.affine_matrix, A)
    np.testing.assert_almost_equal(np.asarray(transform), A)

    coords = np.random.random((20, N + 1)) * 20
    coords[:, -1] = 1
    np.testing.assert_almost_equal(
        (transform @ coords.T).T[:, :-1], transform(coords[:, :-1])
    )


@pytest.mark.parametrize('dimensionality', [2, 3])
def test_affine_matrix_inverse(dimensionality):
    np.random.seed(0)
    N = dimensionality
    A = np.eye(N + 1)
    A[:-1, :-1] = np.random.random((N, N))
    A[:-1, -1] = np.random.random(N)

    # Create transform
    transform = Affine(affine_matrix=A)

    # Check affine was passed correctly
    np.testing.assert_almost_equal(transform.affine_matrix, A)

    # Check inverse is create correctly
    np.testing.assert_almost_equal(
        transform.inverse.affine_matrix, np.linalg.inv(A)
    )


def test_repeat_shear_setting():
    """Test repeatedly setting shear with a lower triangular matrix."""
    # Note this test is needed to check lower triangular
    # decomposition of shear is working
    mat = np.eye(3)
    mat[2, 0] = 0.5
    transform = Affine(shear=mat.copy())
    # Check shear decomposed into lower triangular
    np.testing.assert_almost_equal(mat, transform.shear)

    # Set shear to same value
    transform.shear = mat.copy()
    # Check shear still decomposed into lower triangular
    np.testing.assert_almost_equal(mat, transform.shear)

    # Set shear to same value
    transform.shear = mat.copy()
    # Check shear still decomposed into lower triangular
    np.testing.assert_almost_equal(mat, transform.shear)


@pytest.mark.parametrize('dimensionality', [2, 3])
def test_composite_affine_equiv_to_affine(dimensionality):
    np.random.seed(0)
    translate = np.random.randn(dimensionality)
    scale = np.random.randn(dimensionality)
    rotate = special_ortho_group.rvs(dimensionality)
    shear = np.random.randn((dimensionality * (dimensionality - 1)) // 2)

    composite = CompositeAffine(
        translate=translate, scale=scale, rotate=rotate, shear=shear
    )
    affine = Affine(
        translate=translate, scale=scale, rotate=rotate, shear=shear
    )

    np.testing.assert_almost_equal(
        composite.affine_matrix, affine.affine_matrix
    )
