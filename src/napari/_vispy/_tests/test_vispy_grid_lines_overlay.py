import numpy as np

from napari._vispy.overlays.grid_lines import VispyGridLinesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.visuals.grid_lines import GridLines3D
from napari._vispy.visuals.text import Text
from napari.components import ViewerModel
from napari.components.dims import RangeTuple
from napari.components.overlays import GridLinesOverlay
from napari.utils._units import compute_nice_ticks


def test_scene_axes_dimensions_properly_detected():
    viewer = ViewerModel()
    gridlines_model = GridLinesOverlay()
    gridlines_vispy = VispyGridLinesOverlay(
        viewer=viewer, overlay=gridlines_model, font_info=FontInfo()
    )
    viewer.add_image(
        np.zeros((10, 10, 10)), scale=(1234, 1, 0.002), translate=(-30, 2, 1)
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
            ),
        )


def test_grid_lines_one_tick_visual_per_axis():
    """All ticks of an axis share one Text visual, whatever their number.

    Each visual has its own shader program, and building those is what
    makes creating grid lines overlays slow (e.g. when toggling grid mode).
    """
    grid_lines = GridLines3D(font_info=FontInfo())

    def n_text_visuals():
        return sum(
            isinstance(label, Text)
            for grid in grid_lines.grids
            for label in grid.children
        )

    for n_ticks in (3, 10):
        grid_lines.set_ticks(True, n_ticks, [RangeTuple(0, 100, 1)] * 3)
        # one tick and one axis label visual per axis
        assert n_text_visuals() == 6
        assert len(grid_lines.tick_labels[0].text) > 1
