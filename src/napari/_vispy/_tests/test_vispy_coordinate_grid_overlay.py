import numpy as np
import pytest

from napari._vispy.overlays.coordinate_grid import VispyCoordinateGridOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import CoordinateGridOverlay
from napari.utils._units import compute_nice_ticks


@pytest.mark.parametrize('y_scale', [1, 0.3, 0.05])
def test_scene_axes_dimensions_properly_detected(y_scale):
    viewer = ViewerModel()
    coordinate_grid_model = CoordinateGridOverlay()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=coordinate_grid_model, font_info=FontInfo()
    )
    viewer.add_image(
        np.zeros((10, 10, 10)),
        scale=(1234, y_scale, 0.002),
        translate=(-30, 2, 1),
    )

    viewer.dims.ndisplay = 2
    assert not coordinate_grid_vispy.node.axis_labels[-1].visible

    viewer.dims.ndisplay = 3
    assert coordinate_grid_vispy.node.axis_labels[-1].visible

    for axis in range(3):
        ticks = coordinate_grid_vispy.node.tick_labels[axis]
        range_ = viewer.dims.range[2 - axis]  # zyx -> xyz
        np.testing.assert_array_equal(
            [float(t) for t in ticks.text],
            compute_nice_ticks(
                range_.start, range_.stop, coordinate_grid_model.n_ticks
            )[0],
        )


@pytest.mark.parametrize('ndisplay', [2, 3])
def test_2d_data_draws_single_grid_plane(ndisplay):
    viewer = ViewerModel()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=CoordinateGridOverlay(), font_info=FontInfo()
    )
    viewer.add_image(np.zeros((10, 20)))
    viewer.dims.ndisplay = ndisplay

    grids = coordinate_grid_vispy.node.grids
    # the grid in the plane of the data
    x0, x1, y0, y1 = grids[0].grid_bounds
    assert x1 > x0
    assert y1 > y0
    # the second grid is perpendicular to the data, so it would float
    # outside it when the camera rotates in 3D: it must not draw any area
    x0, x1, y0, y1 = grids[1].grid_bounds
    assert x1 <= x0 or y1 <= y0
    # but it still holds the second axis' labels, which must show
    assert grids[1].visible
    assert coordinate_grid_vispy.node.tick_labels[1].visible
