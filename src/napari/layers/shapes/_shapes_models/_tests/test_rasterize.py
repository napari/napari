import itertools
from contextlib import ExitStack
from unittest import mock

import numpy as np
import numpy.testing as npt
from hypothesis import assume, given, settings, strategies as st
from skimage.draw import line, polygon2mask

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


def reference_to_mask(shape, mask_shape, zoom_factor=1, offset=(0, 0)):
    """Dense rasterization, kept as the oracle for the sparse one."""
    plane = shape.dims_order[-2:]
    embedded = len(mask_shape) != 2
    shape_plane = [mask_shape[d] for d in plane] if embedded else mask_shape
    data = (shape._mask_vertices(plane) - offset) * zoom_factor
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
def test_mask_index_matches_dense_rasterization(
    shape, size, embedded, zoom_factor, offset
):
    ndim = shape.data.shape[1]
    if embedded:
        # zoom and offset are 2D-only, as used by the thumbnail
        mask_shape, zoom_factor, offset = (size + 3,) * ndim, 1, (0, 0)
    else:
        mask_shape = (size, size + 5)
    expected = reference_to_mask(shape, mask_shape, zoom_factor, offset)

    npt.assert_array_equal(
        shape.to_mask(mask_shape, zoom_factor, offset), expected
    )
    labels = np.zeros(mask_shape, dtype=int)
    labels[shape._mask_index(mask_shape, zoom_factor, offset)] = 7
    npt.assert_array_equal(labels == 7, expected)


@settings(max_examples=100, deadline=None)
@given(shape_list=shape_lists(), size=st.integers(5, 40))
def test_to_labels_matches_dense_rasterization(shape_list, size):
    labels_shape = (size,) * shape_list.shapes[0].data.shape[1]
    expected = np.zeros(labels_shape, dtype=int)
    for ind in shape_list._z_order[::-1]:
        expected[reference_to_mask(shape_list.shapes[ind], labels_shape)] = (
            ind + 1
        )
    npt.assert_array_equal(shape_list.to_labels(labels_shape), expected)


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
