"""Tests for napari.components.direction_edge_labels (pure edge-label geometry)."""

import pytest

from napari.components import Camera, Dims, direction_edge_labels
from napari.components._direction_edge_labels import reconcile_direction_labels
from napari.utils.camera_orientations import (
    DepthAxisOrientation,
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)

# A standard axial DICOM-LPS frame reduced to per-world-axis (neg, pos) labels:
# world axis 0 (z, depth) I/S, axis 1 (y, vertical) A/P, axis 2 (x, horizontal)
# R/L. This is the mcs.gui motivating case.
LPS_AXIAL = (('I', 'S'), ('A', 'P'), ('R', 'L'))


def _camera(vertical, horizontal, depth=DepthAxisOrientation.TOWARDS):
    return Camera(orientation=(depth, vertical, horizontal))


UP, DOWN = VerticalAxisOrientation.UP, VerticalAxisOrientation.DOWN
LEFT, RIGHT = HorizontalAxisOrientation.LEFT, HorizontalAxisOrientation.RIGHT


@pytest.mark.parametrize(
    ('vertical', 'horizontal', 'expected'),
    [
        # default: +y (posterior) DOWN -> bottom, +x (left) RIGHT -> right
        (DOWN, RIGHT, {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'}),
        (UP, RIGHT, {'top': 'P', 'bottom': 'A', 'left': 'R', 'right': 'L'}),
        (DOWN, LEFT, {'top': 'A', 'bottom': 'P', 'left': 'L', 'right': 'R'}),
        (UP, LEFT, {'top': 'P', 'bottom': 'A', 'left': 'L', 'right': 'R'}),
    ],
)
def test_camera_orientation_places_each_label_on_its_edge(
    vertical, horizontal, expected
):
    # LPS_AXIAL displayed as (1, 2): axis 1 (A/P) vertical, axis 2 (R/L)
    # horizontal. Each screen axis flips independently with camera.orientation.
    dims = Dims(ndim=3, ndisplay=2)
    cam = _camera(vertical, horizontal)

    assert direction_edge_labels(LPS_AXIAL, dims=dims, camera=cam) == expected


def test_depth_orientation_does_not_affect_in_plane_edges():
    dims = Dims(ndim=3, ndisplay=2)
    toward = direction_edge_labels(
        LPS_AXIAL,
        dims=dims,
        camera=_camera(
            VerticalAxisOrientation.DOWN,
            HorizontalAxisOrientation.RIGHT,
            depth=DepthAxisOrientation.TOWARDS,
        ),
    )
    away = direction_edge_labels(
        LPS_AXIAL,
        dims=dims,
        camera=_camera(
            VerticalAxisOrientation.DOWN,
            HorizontalAxisOrientation.RIGHT,
            depth=DepthAxisOrientation.AWAY,
        ),
    )

    assert toward == away


def test_ndisplay_3_returns_none():
    dims = Dims(ndim=3, ndisplay=3)
    cam = Camera()

    assert direction_edge_labels(LPS_AXIAL, dims=dims, camera=cam) is None


def test_none_labels_return_empty_mapping_in_2d():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()

    assert direction_edge_labels(None, dims=dims, camera=cam) == {}


def test_none_labels_still_return_none_in_3d():
    dims = Dims(ndim=3, ndisplay=3)
    cam = Camera()

    assert direction_edge_labels(None, dims=dims, camera=cam) is None


def test_partial_pair_labels_only_the_present_end():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    # axis 1 labels only its positive end; axis 2 only its negative end.
    labels = (None, (None, 'P'), ('R', None))

    edges = direction_edge_labels(labels, dims=dims, camera=cam)

    assert edges == {'bottom': 'P', 'left': 'R'}


def test_unlabeled_axis_contributes_no_edges():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    labels = (None, None, ('R', 'L'))  # vertical axis unlabeled

    edges = direction_edge_labels(labels, dims=dims, camera=cam)

    assert edges == {'left': 'R', 'right': 'L'}


def test_transposed_order_reassigns_vertical_and_horizontal():
    # order (0, 2, 1) -> displayed == (2, 1): axis 2 vertical, axis 1 horizontal.
    dims = Dims(ndim=3, ndisplay=2, order=(0, 2, 1))
    cam = Camera()

    edges = direction_edge_labels(LPS_AXIAL, dims=dims, camera=cam)

    # axis 2 (R/L) now vertical: +L down -> bottom; axis 1 (A/P) now
    # horizontal: +P right -> right.
    assert edges == {'top': 'R', 'bottom': 'L', 'left': 'A', 'right': 'P'}


def test_four_dimensions_uses_displayed_axes():
    # ndim 4, ndisplay 2 -> displayed == (2, 3).
    dims = Dims(ndim=4, ndisplay=2)
    cam = Camera()
    labels = (None, None, ('R', 'L'), ('A', 'P'))

    edges = direction_edge_labels(labels, dims=dims, camera=cam)

    # axis 2 vertical (R/L), axis 3 horizontal (A/P).
    assert edges == {'top': 'R', 'bottom': 'L', 'left': 'A', 'right': 'P'}


def test_length_mismatch_raises_value_error():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()

    # Use two entries, not one: the message interpolates the label count, and a
    # count != 1 exercises the path that a `{n}` placeholder would crash on
    # (``trans._`` reserves ``n`` for pluralization).
    with pytest.raises(ValueError, match='one entry per dimension'):
        direction_edge_labels((('R', 'L'), ('A', 'P')), dims=dims, camera=cam)


def test_malformed_pair_raises_value_error():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    labels = (None, None, ('R', 'L', 'extra'))

    with pytest.raises(ValueError, match='negative, positive'):
        direction_edge_labels(labels, dims=dims, camera=cam)


def test_malformed_nondisplayed_axis_raises_value_error():
    # axis 0 is not displayed, but a malformed entry there must still raise.
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    labels = (('I', 'S', 'oops'), ('A', 'P'), ('R', 'L'))

    with pytest.raises(ValueError, match='negative, positive'):
        direction_edge_labels(labels, dims=dims, camera=cam)


def test_length_mismatch_validated_even_in_3d():
    # 3D returns None, but a length error must be raised before that.
    dims = Dims(ndim=3, ndisplay=3)
    cam = Camera()

    with pytest.raises(ValueError, match='one entry per dimension'):
        direction_edge_labels((('R', 'L'),), dims=dims, camera=cam)


def test_string_entry_is_rejected_not_unpacked():
    # 'RL' is a 2-length iterable; it must not be accepted as ('R', 'L').
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    labels = (None, None, 'RL')

    with pytest.raises(ValueError, match='negative, positive'):
        direction_edge_labels(labels, dims=dims, camera=cam)


def test_non_string_label_raises_value_error():
    dims = Dims(ndim=3, ndisplay=2)
    cam = Camera()
    labels = (None, None, (1, 2))

    with pytest.raises(ValueError, match='must be a string'):
        direction_edge_labels(labels, dims=dims, camera=cam)


def test_reconcile_pads_leading_axes_with_none():
    labels = (('A', 'P'), ('R', 'L'))
    assert reconcile_direction_labels(labels, 3) == (
        None,
        ('A', 'P'),
        ('R', 'L'),
    )


def test_reconcile_keeps_trailing_when_reducing():
    labels = (('I', 'S'), ('A', 'P'), ('R', 'L'))
    assert reconcile_direction_labels(labels, 2) == (('A', 'P'), ('R', 'L'))


def test_reconcile_exact_length_is_unchanged():
    labels = (('A', 'P'), ('R', 'L'))
    assert reconcile_direction_labels(labels, 2) == labels


def test_reconcile_empty_source_fills_with_none():
    assert reconcile_direction_labels((), 2) == (None, None)


def test_reconcile_to_zero_ndim_is_empty():
    assert reconcile_direction_labels((('A', 'P'),), 0) == ()


def test_reconcile_is_non_destructive_across_ndim_roundtrip():
    # The stored source is a stable suffix frame: reducing then restoring ndim
    # recovers every label, because reconcile always runs from the source.
    stored = (('I', 'S'), ('A', 'P'), ('R', 'L'))
    assert reconcile_direction_labels(stored, 2) == (('A', 'P'), ('R', 'L'))
    assert reconcile_direction_labels(stored, 3) == stored
