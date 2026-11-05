from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.color import Color
from vispy.scene import MatrixTransform, STTransform
from vispy.scene.visuals import GridLines, Node

from napari._vispy.visuals.text import Text
from napari.components.dims import RangeTuple
from napari.utils._units import compute_nice_ticks

if TYPE_CHECKING:
    from vispy.visuals.filters.clipper import Clipper

    from napari._vispy.utils.qt_font import FontInfo
    from napari.utils.color import ColorValue


class CoordinateGrid(Node):
    def __init__(
        self,
        font_info: FontInfo,
    ):
        super().__init__()
        self.font_info = font_info
        self.font_size = 9.0

        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        # one Text visual per axis holds all its ticks: each visual has its
        # own shader program, and building those is slow, notably when
        # overlays are created or moved between views (e.g: grid mode)
        self.tick_labels: list[Text] = [
            Text(font_info=font_info) for _ in range(3)
        ]
        self.axis_labels: list[Text] = [
            Text(font_info=font_info) for _ in range(3)
        ]
        self._color: ColorValue | str = 'white'
        self._opacity = 1.0

        # grids are only created once and then updated in place: creating
        # visuals means building new shader programs, which is slow
        self.grids: list[GridLines] = []
        for axis in range(3):
            grid = GridLines(parent=self, border_width=0, color=self._color)
            grid.transform = MatrixTransform()
            self.grids.append(grid)
            self.axis_labels[axis].transform = STTransform()
            self.axis_labels[axis].parent = grid
            self.tick_labels[axis].transform = STTransform()
            self.tick_labels[axis].parent = grid

    @property
    def color(self) -> ColorValue | str:
        return self._color

    @color.setter
    def color(self, color: ColorValue | str) -> None:
        self._color = color
        # GridLines only takes color on init, but it's stored as a uniform
        # in its shader function, so it can be updated without a rebuild
        for grid in self.grids:
            grid._grid_color_fn['color'] = Color(color).rgba
            grid.update()

    def _set_clipper(self, node: Node, clipper: Clipper | None) -> None:
        """Propagate the viewbox clipper to all child visuals.

        This is just a Node (not VisualNode), so it would otherwise
        ignore the clipper and draw outside its viewbox (like ColorBar).

        Unlike ColorBar, which is a child of the ViewBox, this is a child of
        its scene, which doesn't remove the clipper when the grid moves to
        another viewbox. The base Node does, but only for the clippers it
        finds in `_clippers`, so they need to be stored there.
        """
        if clipper is None:
            self._clippers.pop(node, None)
        else:
            self._clippers[node] = clipper
        # labels are children of the grids, not of this node. We need to explicity
        # set the clipper for all of them otherwise they can get out of sync.
        for grid in self.grids:
            for visual in (grid, *grid.children):
                visual._set_clipper(node, clipper)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    @property
    def opacity(self) -> float:
        return self.grids[0].opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        self._opacity = opacity
        for grid in self.grids:
            grid.opacity = opacity
        for ticks in self.tick_labels:
            ticks.opacity = opacity
        for axis in self.axis_labels:
            axis.opacity = opacity

    def set_extents(self, ranges: list[RangeTuple]) -> None:
        ndisplay = len(ranges)
        bounds = []
        for i in range(ndisplay):
            rng0 = ranges[i]
            rng1 = ranges[(i + 1) % ndisplay]
            bounds.append((rng0.start, rng0.stop, rng1.start, rng1.stop))
            self.axis_labels[i].pos = (rng0.stop, 0)
            self.axis_labels[i].text = str(i)

        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound

        is_3d = len(bounds) == 3
        self.grids[2].visible = is_3d
        self.tick_labels[2].visible = is_3d
        self.axis_labels[2].visible = is_3d

        self.update()

    def set_view_direction(
        self,
        ranges: tuple[RangeTuple, RangeTuple, RangeTuple],
        view_is_flipped: tuple[bool, bool, bool],
        zoom: float,
    ) -> None:

        # translate everything with the grid on xy plane, we transpose later
        far_bounds = []
        for axis in range(3):
            self.grids[axis].transform.reset()
            self.tick_labels[axis].transform.translate = (0, 0, 0)
            self.axis_labels[axis].transform.translate = (0, 0, 0)

            # get the translation necessary to bring the grid to the back
            # and also the position of the near bound for later
            far_bound_idx = int(view_is_flipped[axis])
            farthest_bound = ranges[axis][far_bound_idx]
            far_bounds.append(farthest_bound)

        # magic numbers, they work ok at a wide range of zooms
        tick_offset = self.font_size * 2 / zoom

        # offset ticks for each axis so they are positioned nicely for readability,
        # putting them all on the outside of the volume (over the background)
        for axis in range(3):
            prev_axis = (axis - 1) % 3
            # move grid to the back of the whole volume (far bound)
            self.grids[axis].transform.translate((0, 0, far_bounds[prev_axis]))

            # shift according to view angle to maximize visibility and have consistent positioning.
            # These branches were found by trial and error with the goal to reproduce the
            # tick positioning by plotly (e.g: https://plotly.com/python/3d-scatter-plots/)

            self._translate_text_based_on_camera(
                self.tick_labels[axis],
                axis,
                far_bounds,
                ranges,
                tick_offset,
                view_is_flipped,
            )
            # offset * 3 for axis labels so they don't overlap
            self._translate_text_based_on_camera(
                self.axis_labels[axis],
                axis,
                far_bounds,
                ranges,
                tick_offset * 3,
                view_is_flipped,
            )

        # rotate grids onto the right axes
        for axis in range(3):
            self.grids[axis].transform.rotate(angle=120 * axis, axis=(1, 1, 1))

    def _translate_text_based_on_camera(
        self,
        text: Text,
        axis: int,
        far_bounds: list[float],
        ranges: tuple[RangeTuple, ...],
        offset: float,
        view_is_flipped: tuple[bool, ...],
    ) -> None:
        prev_axis = (axis - 1) % 3
        next_axis = (axis + 1) % 3

        start_next = ranges[next_axis][0]
        stop_next = ranges[next_axis][1]

        start_prev = ranges[prev_axis][0]
        stop_prev = ranges[prev_axis][1]

        # undo shifts caused by grid transform so we're back to the
        # reference frame of the axes (since ticks are not necessarily
        # on the same side as the grid lines depending on view)
        text.transform.move((0, 0, -far_bounds[prev_axis]))

        if view_is_flipped[next_axis] ^ view_is_flipped[prev_axis]:
            text.transform.move((0, -offset, start_prev - offset))
        else:
            if axis == 0:
                # special case one axis so it's visually nicer (all axes are
                # on the "outside" and all are on the "same side")
                text.transform.move((0, -offset, stop_prev + offset))
            else:
                text.transform.move(
                    (0, stop_next - start_next + offset, start_prev - offset)
                )

    def set_ticks(
        self,
        show_ticks: bool,
        n_ticks: int,
        ranges: list[RangeTuple],
    ) -> None:
        if not show_ticks:
            for ticks in self.tick_labels:
                ticks.visible = False
            return

        ndim = len(ranges)
        for axis in range(ndim):
            next_axis = (axis + 1) % ndim
            ticks = self.tick_labels[axis]
            # generate tick positions with round values
            values, decimals = compute_nice_ticks(
                ranges[axis].start, ranges[axis].stop, target_ticks=n_ticks
            )
            text = []
            pos = []
            for val in values:
                text.append(
                    np.format_float_positional(val, decimals, trim='-')
                )
                pos.append((val, ranges[next_axis].start, 0))
            ticks.text = text
            ticks.pos = pos
            ticks.color = self.color
            ticks.opacity = self._opacity
            ticks.visible = len(values) > 0
            ticks.font_size = self.font_size

    def set_axis_labels(
        self, show_labels: bool, ranges: list[RangeTuple], labels: list[str]
    ) -> None:
        ndim = len(ranges)
        for axis in range(ndim):
            next_axis = (axis + 1) % ndim
            visual = self.axis_labels[axis]
            visual.text = labels[axis]
            visual.pos = (
                (ranges[axis].start + ranges[axis].stop) / 2,
                ranges[next_axis].start,
                0,
            )
            visual.color = self.color
            visual.opacity = self._opacity
            visual.font_size = self.font_size * 1.5
            visual.visible = show_labels
