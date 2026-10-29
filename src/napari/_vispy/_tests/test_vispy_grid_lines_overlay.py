import numpy as np
import pytest

from napari._vispy.overlays.grid_lines import VispyGridLinesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import GridLinesOverlay
from napari.utils._units import compute_nice_ticks


@pytest.mark.parametrize('y_scale', [1, 0.3, 0.05])
def test_scene_axes_dimensions_properly_detected(y_scale):
    viewer = ViewerModel()
    gridlines_model = GridLinesOverlay()
    gridlines_vispy = VispyGridLinesOverlay(
        viewer=viewer, overlay=gridlines_model, font_info=FontInfo()
    )
    viewer.add_image(
        np.zeros((10, 10, 10)),
        scale=(1234, y_scale, 0.002),
        translate=(-30, 2, 1),
    )

    viewer.dims.ndisplay = 2
    assert not gridlines_vispy.node.axis_labels[-1].visible

    viewer.dims.ndisplay = 3
    assert gridlines_vispy.node.axis_labels[-1].visible

    for axis in range(3):
        ticks = gridlines_vispy.node.tick_labels[axis]
        range_ = viewer.dims.range[2 - axis]  # zyx -> xyz
        np.testing.assert_array_equal(
            [float(t) for t in ticks.text],
            compute_nice_ticks(
                range_.start, range_.stop, gridlines_model.n_ticks
            )[0],
        )
