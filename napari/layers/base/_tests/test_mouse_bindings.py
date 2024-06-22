from unittest.mock import Mock

import numpy as np

from napari.layers.base._base_mouse_bindings import _rotate_with_box
from napari.utils.transforms import Affine


def test_interaction_box_rotation():
    layer = Mock(affine=Affine())
    initial_affine = Affine()
    initial_mouse_pos = Mock()
    # rotation handle is 8th
    initial_handle_coords = np.asarray(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [6, 3],
        ],
        dtype=np.float32,
    )
    initial_center = np.asarray([3, 3], dtype=np.float32)
    mouse_pos = np.asarray([6, 5], dtype=np.float32)
    event = Mock(dims_displayed=[0, 1], modifiers=[None])
    _rotate_with_box(
        layer,
        initial_affine,
        initial_mouse_pos,
        initial_handle_coords,
        initial_center,
        mouse_pos,
        event,
    )
    # should be ~33 degrees
    assert np.allclose(layer.affine.rotate, Affine(rotate=33.69).rotate)


def test_interaction_box_fixed_rotation():
    layer = Mock(affine=Affine())
    initial_affine = Affine()
    initial_mouse_pos = Mock()
    # rotation handle is 8th
    initial_handle_coords = np.asarray(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [6, 3],
        ],
        dtype=np.float32,
    )
    initial_center = np.asarray([3, 3], dtype=np.float32)
    mouse_pos = np.asarray([6, 5], dtype=np.float32)
    # use Shift to snap rotation to steps of 45 degrees
    event = Mock(dims_displayed=[0, 1], modifiers=['Shift'])
    _rotate_with_box(
        layer,
        initial_affine,
        initial_mouse_pos,
        initial_handle_coords,
        initial_center,
        mouse_pos,
        event,
    )
    # should be 45 degrees
    assert np.allclose(
        layer.affine.rotate, Affine(rotate=45).rotate
    )  # now lets use shift to fix
