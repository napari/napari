from unittest.mock import Mock

import numpy as np
import pytest

from napari.layers.base._base_constants import InteractionBoxHandle
from napari.layers.base._base_mouse_bindings import (
    _rotate_with_box,
    _scale_with_box,
    _translate_with_box,
)
from napari.utils.transforms import Affine


@pytest.mark.parametrize('dims_displayed', [[0, 1], [1, 2]])
def test_interaction_box_translation(dims_displayed):
    layer = Mock(affine=Affine())
    layer._slice_input.displayed = [0, 1]
    initial_affine = Affine()
    initial_mouse_pos = np.asarray([3, 3], dtype=np.float32)
    mouse_pos = np.asarray([6, 5], dtype=np.float32)
    event = Mock(dims_displayed=dims_displayed, modifiers=[None])
    _translate_with_box(
        layer,
        initial_affine,
        initial_mouse_pos,
        mouse_pos,
        event,
    )
    # translate should be equal to [3, 2] from doing [6, 5] - [3, 3]
    assert np.array_equal(
        layer.affine.translate,
        Affine(translate=np.asarray([3, 2], dtype=np.float32)).translate,
    )


@pytest.mark.parametrize('dims_displayed', [[0, 1], [1, 2]])
def test_interaction_box_rotation(dims_displayed):
    layer = Mock(affine=Affine())
    layer._slice_input.displayed = [0, 1]
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
    event = Mock(dims_displayed=dims_displayed, modifiers=[None])
    _rotate_with_box(
        layer,
        initial_affine,
        initial_mouse_pos,
        initial_handle_coords,
        initial_center,
        mouse_pos,
        event,
    )
    # should be approximately 33 degrees
    assert np.allclose(layer.affine.rotate, Affine(rotate=33.69).rotate)


@pytest.mark.parametrize('dims_displayed', [[0, 1], [1, 2]])
def test_interaction_box_fixed_rotation(dims_displayed):
    layer = Mock(affine=Affine())
    layer._slice_input.displayed = [0, 1]
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
    assert np.allclose(layer.affine.rotate, Affine(rotate=45).rotate)


@pytest.mark.parametrize('dims_displayed', [[0, 1], [1, 2]])
def test_interaction_box_scale_with_fixed_aspect(dims_displayed):
    layer = Mock(affine=Affine())
    layer._slice_input.displayed = [0, 1]
    initial_handle_coords_data = np.asarray(
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
    initial_affine = Affine()
    initial_world_to_data = Affine()
    initial_data2physical = Affine()
    nearby_handle = InteractionBoxHandle.TOP_LEFT
    initial_center = np.asarray([3, 3], dtype=np.float32)
    mouse_pos = np.asarray([0, 0], dtype=np.float32)
    event = Mock(dims_displayed=dims_displayed, modifiers=['Shift'])
    _scale_with_box(
        layer,
        initial_affine,
        initial_world_to_data,
        initial_data2physical,
        nearby_handle,
        initial_center,
        initial_handle_coords_data,
        mouse_pos,
        event,
    )
    # when clicking on handle, scale should be 1
    assert np.allclose(
        layer.affine.scale,
        Affine(scale=np.asarray([1, 1], dtype=np.float32)).scale,
    )
