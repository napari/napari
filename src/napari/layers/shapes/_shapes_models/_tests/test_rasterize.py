import itertools
from contextlib import ExitStack
from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, given, settings, strategies as st
from skimage.draw import line, polygon2mask
from skimage.measure import label

from napari.layers.shapes._shape_list import ShapeList
from napari.layers.shapes._shapes_models import (
    Ellipse,
    Line,
    Path,
    Polygon,
    Rectangle,
)
from napari.utils.misc import argsort

N_VERTICES = {Polygon: (3, 6), Path: (2, 6), Line: (2, 2)}


def dense_path_mask(mask_shape, vertices):
    """Path rasterization as it was before the sparse rewrite."""
    mask_shape = np.asarray(mask_shape, dtype=int)
    mask = np.zeros(mask_shape, dtype=bool)
    vertices = np.round(np.clip(vertices, 0, mask_shape - 1)).astype(int)
    duplicates = np.all(np.diff(vertices, axis=0) == 0, axis=-1)
    vertices = vertices[~np.concatenate(([False], duplicates))]
    iis, jjs = [], []
    for v1, v2 in itertools.pairwise(vertices):
        ii, jj = line(*v1, *v2)
        iis.extend(ii.tolist())
        jjs.extend(jj.tolist())
    mask[iis, jjs] = 1
    return mask


def reference_to_mask(
    shape, mask_shape, zoom_factor=1, offset=(0, 0), data_order=False
):
    """Dense rasterization, kept as the oracle for the sparse one."""
    plane = shape.dims_order[-2:]
    embedded = data_order or len(mask_shape) != 2
    shape_plane = [mask_shape[d] for d in plane] if embedded else mask_shape
    data = (shape._vertices_for_mask(plane) - offset) * zoom_factor
    to_mask = polygon2mask if shape._filled else dense_path_mask
    mask_p = to_mask(shape_plane, data)
    if not embedded:
        return mask_p
    mask = np.zeros(mask_shape, dtype=bool)
    others = shape.dims_order[:-2]
    others_key = shape._slice_key_of(others)
    key = [slice(None)] * len(mask_shape)
    for col, dim in enumerate(others):
        key[dim] = slice(others_key[0, col], others_key[1, col] + 1)
    mask[tuple(key)] = np.expand_dims(
        mask_p.transpose(argsort(plane)), tuple(others)
    )
    return mask


def leaves_plane(shape):
    """Lines and paths not confined to their 2D plane are drawn in nD."""
    key = shape._slice_key_of(shape.dims_order[:-2])
    return not shape._filled and bool((key[0] != key[1]).any())


def plane_vertices(draw, shape_class):
    """Well formed 2D vertices, so construction never fails triangulation."""
    center = np.array(draw(st.tuples(*[st.floats(-10, 40)] * 2)))
    if shape_class in (Path, Line):
        n = draw(st.integers(*N_VERTICES[shape_class]))
        steps = draw(
            st.lists(
                st.tuples(*[st.floats(-12, 12)] * 2), min_size=n, max_size=n
            )
        )
        return center + np.cumsum(steps, axis=0)
    if shape_class is Polygon:
        # star-shaped, so never self-intersecting
        n = draw(st.integers(*N_VERTICES[Polygon]))
        angles = np.sort(
            draw(
                st.lists(
                    st.floats(0, 2 * np.pi),
                    min_size=n,
                    max_size=n,
                    unique=True,
                )
            )
        )
        assume(np.all(np.diff(angles) > 0.2))
        radii = np.array(
            draw(st.lists(st.floats(1, 15), min_size=n, max_size=n))
        )
        return center + radii[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
    half = np.array(draw(st.tuples(*[st.floats(0.5, 15)] * 2)))
    theta = draw(st.floats(0, np.pi))
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * half
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return center + corners @ rot.T


def draw_layout(draw, max_ndim):
    ndim = draw(st.integers(2, max_ndim))
    dims_order = list(draw(st.permutations(range(ndim))))
    ndisplay = 3 if ndim > 2 and draw(st.booleans()) else 2
    return ndim, dims_order, ndisplay


def build_shape(draw, ndim, dims_order, ndisplay, z_index=0):
    shape_class = draw(
        st.sampled_from([Polygon, Rectangle, Ellipse, Path, Line])
    )
    in_plane = plane_vertices(draw, shape_class)
    vertices = np.empty((len(in_plane), ndim))
    vertices[:, dims_order[-2:]] = in_plane
    for dim in dims_order[:-2]:
        if draw(st.booleans()):
            # drawn in a 2D view: constant off-plane coordinate
            vertices[:, dim] = draw(st.floats(-3, 12))
        else:
            vertices[:, dim] = draw(
                st.lists(
                    st.floats(-3, 12),
                    min_size=len(in_plane),
                    max_size=len(in_plane),
                )
            )
    try:
        with ExitStack() as stack:
            for module in ('_shapes_utils', '_shapes_models.shape'):
                # no debug dumps from the triangulation failures skipped below
                stack.enter_context(
                    mock.patch(
                        f'napari.layers.shapes.{module}._save_failed_triangulation',
                        return_value=('', ''),
                    )
                )
            shape = shape_class(
                vertices, dims_order=dims_order, z_index=z_index
            )
            shape.ndisplay = ndisplay
    except RuntimeError:
        # vispy occasionally fails to triangulate a valid polygon (about 1 in
        # 1000 here); rasterization is what is under test, so skip those
        assume(False)
    return shape


@st.composite
def shapes(draw):
    return build_shape(draw, *draw_layout(draw, max_ndim=4))


@st.composite
def shape_lists(draw):
    ndim, dims_order, ndisplay = draw_layout(draw, max_ndim=3)
    shape_list = ShapeList(ndisplay=ndisplay)
    color = st.lists(st.floats(0, 1), min_size=4, max_size=4)
    for _ in range(draw(st.integers(1, 4))):
        z_index = draw(st.integers(-2, 2))
        shape_list.add(
            build_shape(draw, ndim, dims_order, ndisplay, z_index),
            face_color=np.array(draw(color)),
            edge_color=np.array(draw(color)),
        )
    return shape_list


zoom_factors = st.floats(0.1, 3)
offsets = st.tuples(st.floats(-10, 10), st.floats(-10, 10))


@settings(max_examples=200, deadline=None)
@given(
    shape=shapes(),
    size=st.integers(1, 40),
    embedded=st.booleans(),
    zoom_factor=zoom_factors,
    offset=offsets,
)
def test_indices_match_dense_rasterization(
    shape, size, embedded, zoom_factor, offset
):
    ndim = shape.data.shape[1]
    if embedded:
        # a mask with every data dimension; zoom and offset are for thumbnails
        mask_shape = (size + 3,) * ndim
        assume(not leaves_plane(shape))
        index = shape._data_index(mask_shape)
        expected = reference_to_mask(shape, mask_shape, data_order=True)
    else:
        mask_shape = (size, size + 5)
        index = shape._display_index(mask_shape, zoom_factor, offset)
        expected = reference_to_mask(shape, mask_shape, zoom_factor, offset)
        npt.assert_array_equal(
            shape.to_mask(mask_shape, zoom_factor, offset), expected
        )
    labels = np.zeros(mask_shape, dtype=int)
    labels[index] = 7
    npt.assert_array_equal(labels == 7, expected)
    if embedded and ndim > 2:
        npt.assert_array_equal(shape.to_mask(mask_shape), expected)


@settings(max_examples=100, deadline=None)
@given(shape_list=shape_lists(), size=st.integers(5, 40))
def test_labels_and_masks_match_dense_rasterization(shape_list, size):
    assume(not any(map(leaves_plane, shape_list.shapes)))
    labels_shape = (size,) * shape_list.shapes[0].data.shape[1]
    masks = np.array(
        [
            reference_to_mask(s, labels_shape, data_order=True)
            for s in shape_list.shapes
        ]
    )
    expected = np.zeros(labels_shape, dtype=int)
    for ind in shape_list._z_order[::-1]:
        expected[masks[ind]] = ind + 1

    npt.assert_array_equal(shape_list.to_labels(labels_shape), expected)
    result = shape_list.to_masks(labels_shape)
    assert result.dtype == masks.dtype
    npt.assert_array_equal(result, masks)


def test_to_masks_of_empty_list_is_unchanged():
    masks = ShapeList().to_masks((4, 5))
    assert masks.shape == (0,)
    assert masks.dtype == np.float64


@settings(max_examples=100, deadline=None)
@given(shape_list=shape_lists(), zoom_factor=zoom_factors, offset=offsets)
def test_to_colors_matches_dense_rasterization(
    shape_list, zoom_factor, offset
):
    expected = np.zeros((30, 40, 4))
    expected[..., 3] = 1
    in_view = np.isin(shape_list._z_order, np.argwhere(shape_list._displayed))
    for ind in shape_list._z_order[in_view]:
        shape = shape_list.shapes[ind]
        mask = reference_to_mask(shape, (30, 40), zoom_factor, offset)
        is_path = isinstance(shape, (Path, Line))
        colors = shape_list._edge_color if is_path else shape_list._face_color
        expected[mask, :] = colors[ind]
    npt.assert_array_equal(
        shape_list.to_colors((30, 40), zoom_factor, offset), expected
    )


PATH_3D = np.array(
    [
        [0, 0, 0],
        [0, 10, 10],
        [0, 5, 15],
        [20, 5, 15],
        [56, 70, 21],
        [127, 127, 127],
    ]
)


@pytest.mark.parametrize('ndisplay', [2, 3])
@pytest.mark.parametrize(
    ('shape_class', 'vertices'),
    [(Path, PATH_3D), (Line, PATH_3D[[1, -1]])],
)
def test_path_leaving_its_plane_is_drawn_as_nd_line(
    shape_class, vertices, ndisplay
):
    shape = shape_class(vertices)
    shape.ndisplay = ndisplay
    mask = shape.to_mask((128, 128, 128))

    assert all(mask[tuple(v)] for v in vertices)
    assert label(mask, connectivity=3).max() == 1
    # one voxel per step along the longest axis of each segment, not a prism
    steps = np.abs(np.diff(vertices, axis=0)).max(axis=1).sum()
    assert mask.sum() <= steps + 1


def test_polygon_leaving_its_plane_keeps_prism_rasterization():
    polygon = Polygon(
        np.array([[0, 0, 0], [0, 0, 10], [4, 10, 10], [4, 10, 0]])
    )
    mask = polygon.to_mask((5, 11, 11))
    npt.assert_array_equal(mask, reference_to_mask(polygon, (5, 11, 11)))
    assert mask.any(axis=(1, 2)).all()


@pytest.mark.parametrize('origin', [(0, 0, 0), (2, -3, 4), (-5, 6, -1)])
def test_to_labels_origin_is_a_shifted_window(origin):
    # shapes stay inside every window: paths are clamped at the border
    shape_list = ShapeList(
        [
            Polygon(np.array([[8, 6, 6], [8, 6, 13], [8, 13, 10]])),
            Path(np.array([[6, 7, 13], [12, 12, 6], [12, 13, 13]])),
        ]
    )
    full = shape_list.to_labels((40, 40, 40), origin=np.array([-10] * 3))
    window = tuple(slice(o + 10, o + 30) for o in origin)
    npt.assert_array_equal(
        shape_list.to_labels((20, 20, 20), origin=np.array(origin)),
        full[window],
    )


def test_labels_and_masks_of_2d_shapes_ignore_rolled_dims():
    rect = np.array([[2, 1], [2, 3], [12, 3], [12, 1]])
    expected = ShapeList()
    expected.add(Rectangle(rect))
    rolled = ShapeList()
    rolled.add(Rectangle(rect, dims_order=[1, 0]))
    npt.assert_array_equal(
        rolled.to_labels((15, 20)), expected.to_labels((15, 20))
    )
    npt.assert_array_equal(
        rolled.to_masks((15, 20)), expected.to_masks((15, 20))
    )
