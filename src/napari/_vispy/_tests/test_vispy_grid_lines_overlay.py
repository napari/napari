import numpy as np
from vispy.scene import ViewBox

from napari._vispy.overlays.grid_lines import VispyGridLinesOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari._vispy.visuals.grid_lines import GridLines3D
from napari._vispy.visuals.text import Text
from napari.components import ViewerModel
from napari.components.dims import RangeTuple
from napari.components.overlays import GridLinesOverlay
from napari.utils._units import compute_nice_ticks
from napari.utils.colormaps.standardize_color import transform_color


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


def test_grid_lines_clipped_to_viewbox():
    """Grid lines must not draw outside their viewbox (#9807)."""
    grid_lines = GridLines3D(font_info=FontInfo())
    grid_lines.set_ticks(True, 3, [RangeTuple(0, 10, 1)] * 3)
    # the visuals that draw: grids, with axis and tick labels as children
    visuals = [v for grid in grid_lines.grids for v in (grid, *grid.children)]

    def clipped_to(view):
        return [view.scene.clipper in v._vshare.filters for v in visuals]

    view_1, view_2 = ViewBox(), ViewBox()
    grid_lines.parent = view_1.scene
    assert all(clipped_to(view_1))

    # toggling grid mode can move the overlay to another viewbox
    grid_lines.parent = view_2.scene
    assert all(clipped_to(view_2))
    assert not any(clipped_to(view_1))


def test_grid_lines_updated_in_place():
    """Grid visuals are reused, not recreated, when the view changes.

    Creating visuals means building new shader programs, which is slow.
    """
    viewer = ViewerModel()
    viewer.add_image(np.zeros((10, 10, 10)))
    overlay = GridLinesOverlay(visible=True)
    vispy_overlay = VispyGridLinesOverlay(
        viewer=viewer, overlay=overlay, font_info=FontInfo()
    )
    grid_lines = vispy_overlay.node
    grids = list(grid_lines.grids)

    viewer.dims.ndisplay = 3
    viewer.dims.order = (1, 0, 2)
    overlay.color = 'red'
    assert grid_lines.grids == grids

    for grid in grid_lines.grids:
        color = grid._grid_color_fn['color'].value
        np.testing.assert_array_equal(color, transform_color('red')[0])
