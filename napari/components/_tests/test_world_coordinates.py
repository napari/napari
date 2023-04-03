import warnings

import numpy as np
import pytest

from napari.components import ViewerModel


def test_translated_images():
    """Test two translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data, translate=[10, 0, 0])
    assert viewer.dims.range[0] == (0, 20, 1)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == (20, 10, 10)
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.point_slider[0] == i


def test_scaled_images():
    """Test two scaled images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data[::2], scale=[2, 1, 1])
    # TODO: non-integer with mixed scale?
    assert viewer.dims.range[0] == (-0.5, 10, 1)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == (10, 10, 10)
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.point_slider[0] == i


def test_scaled_and_translated_images():
    """Test scaled and translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data)
    viewer.add_image(data[::2], scale=[2, 1, 1], translate=[10, 0, 0])
    # TODO: non-integer with mixed scale?
    assert viewer.dims.range[0] == (
        0,
        19.5,
        1,
    )
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == (19, 10, 10)
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.point_slider[0] == i


def test_both_scaled_and_translated_images():
    """Test both scaled and translated images."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data, scale=[2, 1, 1])
    viewer.add_image(data, scale=[2, 1, 1], translate=[20, 0, 0])
    assert viewer.dims.range[0] == (0, 40, 2)
    assert viewer.dims.range[1] == (0, 10, 1)
    assert viewer.dims.range[2] == (0, 10, 1)
    assert viewer.dims.nsteps == (20, 10, 10)
    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.set_current_step(0, i)
        assert viewer.dims.point_slider[0] == i


def test_no_warning_non_affine_slicing():
    """Test no warning if not slicing into an affine."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    viewer.add_image(data, scale=[2, 1, 1], translate=[10, 15, 20])
    with warnings.catch_warnings(record=True) as recorded_warnings:
        viewer.layers[0].refresh()
    assert len(recorded_warnings) == 0


def test_warning_affine_slicing():
    """Test warning if slicing into an affine."""
    viewer = ViewerModel()
    np.random.seed(0)
    data = np.random.random((10, 10, 10))
    with pytest.warns(UserWarning) as wrn:
        viewer.add_image(
            data,
            scale=[2, 1, 1],
            translate=[10, 15, 20],
            shear=[[1, 0, 0], [0, 1, 0], [4, 0, 1]],
        )
    assert 'Non-orthogonal slicing is being requested' in str(wrn[0].message)
    with pytest.warns(UserWarning) as recorded_warnings:
        viewer.layers[0].refresh()
    assert len(recorded_warnings) == 1
