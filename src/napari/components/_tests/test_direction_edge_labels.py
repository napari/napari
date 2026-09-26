import pytest

from napari.components import Camera, Dims, direction_edge_labels
from napari.utils.camera_orientations import (
    DepthAxisOrientation,
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)

# (negative, positive) labels for world axes (z, y, x) of an axial DICOM LPS
# image.
LPS_AXIAL = (('I', 'S'), ('A', 'P'), ('R', 'L'))

UP, DOWN = VerticalAxisOrientation.UP, VerticalAxisOrientation.DOWN
LEFT, RIGHT = HorizontalAxisOrientation.LEFT, HorizontalAxisOrientation.RIGHT


@pytest.mark.parametrize('depth', list(DepthAxisOrientation))
@pytest.mark.parametrize(
    ('vertical', 'horizontal', 'expected'),
    [
        (DOWN, RIGHT, {'top': 'A', 'bottom': 'P', 'left': 'R', 'right': 'L'}),
        (UP, RIGHT, {'top': 'P', 'bottom': 'A', 'left': 'R', 'right': 'L'}),
        (DOWN, LEFT, {'top': 'A', 'bottom': 'P', 'left': 'L', 'right': 'R'}),
        (UP, LEFT, {'top': 'P', 'bottom': 'A', 'left': 'L', 'right': 'R'}),
    ],
)
def test_camera_orientation_places_labels(
    depth, vertical, horizontal, expected
):
    camera = Camera(orientation=(depth, vertical, horizontal))

    edges = direction_edge_labels(
        LPS_AXIAL, dims=Dims(ndim=3, ndisplay=2), camera=camera
    )

    assert edges == expected


@pytest.mark.parametrize(
    ('dims', 'labels', 'expected'),
    [
        # transposed: axis 2 is vertical, axis 1 horizontal
        (
            Dims(ndim=3, ndisplay=2, order=(0, 2, 1)),
            LPS_AXIAL,
            {'top': 'R', 'bottom': 'L', 'left': 'A', 'right': 'P'},
        ),
        (
            Dims(ndim=4, ndisplay=2),
            (None, None, ('R', 'L'), ('A', 'P')),
            {'top': 'R', 'bottom': 'L', 'left': 'A', 'right': 'P'},
        ),
        (
            Dims(ndim=3, ndisplay=2),
            (None, (None, 'P'), ('R', None)),
            {'bottom': 'P', 'left': 'R'},
        ),
        (
            Dims(ndim=3, ndisplay=2),
            (None, None, ('R', 'L')),
            {'left': 'R', 'right': 'L'},
        ),
        (Dims(ndim=3, ndisplay=2), (None, None, None), {}),
    ],
)
def test_displayed_axes_and_unlabeled_ends(dims, labels, expected):
    assert (
        direction_edge_labels(labels, dims=dims, camera=Camera()) == expected
    )


@pytest.mark.parametrize(
    'dims',
    [
        Dims(ndim=3, ndisplay=3),
        # displays two axes, but in 3D
        Dims(ndim=2, ndisplay=3),
        Dims(ndim=1, ndisplay=2),
        Dims(ndim=0, ndisplay=2),
    ],
)
def test_undefined_mapping_returns_none(dims):
    labels = LPS_AXIAL[3 - dims.ndim :]

    assert direction_edge_labels(labels, dims=dims, camera=Camera()) is None


@pytest.mark.parametrize('ndisplay', [2, 3])
def test_length_mismatch_raises(ndisplay):
    with pytest.raises(ValueError, match='one entry per dimension'):
        direction_edge_labels(
            LPS_AXIAL[1:],
            dims=Dims(ndim=3, ndisplay=ndisplay),
            camera=Camera(),
        )
