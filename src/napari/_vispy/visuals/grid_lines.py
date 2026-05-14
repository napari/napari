from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene import MatrixTransform, STTransform
from vispy.scene.visuals import GridLines, Node, Text

from napari.components.dims import RangeTuple
from napari.utils._units import compute_nice_ticks

if TYPE_CHECKING:
    from vispy.visuals.text.text import FontManager

    from napari.utils.color import ColorValue


class GridLines3D(Node):
    def __init__(
        self,
        font_manager: FontManager | None = None,
        font_family: str = 'OpenSans',
    ):
        super().__init__()
        self.font_manager = font_manager
        self.font_family = font_family

        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        self.tick_labels: dict[int, list[Text]] = {0: [], 1: [], 2: []}
        self._last_ticks = np.array((0.0,))
        self._last_view_is_flipped: tuple[bool, ...] = ()
        self._last_up_direction: tuple[int, ...] = ()
        self._last_orientation_flip: tuple[int, ...] = ()
        self.color: ColorValue | str = 'white'
        self.scale = (1, 1, 1)
        self._opacity = 1.0

        self.grids: list[GridLines] = []
        self.reset_grids()

    def reset_grids(self) -> None:
        # color and scale are not exposed on the visual, so this is how we set them...
        for grid in self.grids:
            grid.parent = None
        self.grids.clear()

        for _ in range(3):
            grid = GridLines(
                parent=self,
                border_width=0,
                color=self.color,
                scale=self.scale,
            )
            grid.transform = MatrixTransform()
            grid.opacity = self._opacity
            self.grids.append(grid)

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
        for ticks in self.tick_labels.values():
            for tick in ticks:
                tick.opacity = opacity

    def set_extents(self, ranges: list[RangeTuple]) -> None:
        ndisplay = len(ranges)
        bounds = []
        for i in range(ndisplay):
            rng0 = ranges[i]
            rng1 = ranges[(i + 1) % ndisplay]
            bounds.append((rng0.start, rng0.stop, rng1.start, rng1.stop))

        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound

        is_3d = len(bounds) == 3
        self.grids[2].visible = is_3d
        for tick in self.tick_labels[2]:
            tick.visible = is_3d

        self.update()

    def set_view_direction(
        self,
        ranges: tuple[RangeTuple],
        view_direction: tuple[int, ...],
        up_direction: tuple[int, ...],
        orientation_flip: tuple[int, ...],
        force: bool = False,
    ) -> None:
        view_is_flipped = tuple(
            d * f >= 0
            for d, f in zip(view_direction, orientation_flip, strict=False)
        )

        if len(ranges) == 2:
            ranges = ranges + ((0, 0),)
            view_is_flipped = view_is_flipped + (False,)

        if not force and (
            np.array_equal(view_is_flipped, self._last_view_is_flipped)
            and np.array_equal(up_direction, self._last_up_direction)
            and np.array_equal(orientation_flip, self._last_orientation_flip)
        ):
            return

        # translate everything with the grid on xy plane, we transpose later
        far_bounds = []
        for axis in range(3):
            self.grids[axis].transform.reset()
            for tick in self.tick_labels[axis]:
                tick.transform.translate = (0, 0, 0)

            # get the translation necessary to bring the grid to the back
            farthest_bound = ranges[axis][int(view_is_flipped[axis])]
            far_bounds.append(farthest_bound)

        for axis in range(3):
            prev_axis = (axis - 1) % 3
            next_axis = (axis + 1) % 3
            self.grids[axis].transform.translate((0, 0, far_bounds[prev_axis]))

            for tick in self.tick_labels[axis]:
                # undo shifts caused by grid transform so we're back to the axes
                tick.transform.move((0, 0, -far_bounds[prev_axis]))

                # shift according to view angle to maximize visibility and have consistent positioning
                # these branches were found by trial and error with the goal to reproduce the
                # tick positioning by plotly (e.g: https://plotly.com/python/3d-scatter-plots/)
                next_axis_shift = ranges[next_axis][1] - ranges[next_axis][0]
                prev_axis_shift = ranges[prev_axis][1] - ranges[prev_axis][0]

                if axis == 0:
                    anchor_flip = -1
                    if not view_is_flipped[next_axis]:
                        tick.transform.move((0, next_axis_shift, 0))
                        anchor_flip *= -1
                    if view_is_flipped[prev_axis]:
                        tick.transform.move((0, 0, prev_axis_shift))
                        anchor_flip *= -1
                if axis == 1:
                    anchor_flip = 1
                    if view_is_flipped[next_axis]:
                        tick.transform.move((0, next_axis_shift, 0))
                        anchor_flip *= -1
                    if not view_is_flipped[prev_axis]:
                        tick.transform.move((0, 0, prev_axis_shift))
                        anchor_flip *= -1
                if axis == 2:
                    anchor_flip = -1
                    if not view_is_flipped[next_axis]:
                        tick.transform.move((0, 0, prev_axis_shift))
                    if view_is_flipped[prev_axis]:
                        tick.transform.move((0, next_axis_shift, 0))

                # this is just black magic at this point... but hey, it works
                if (
                    up_direction[axis]
                    * anchor_flip
                    * orientation_flip[next_axis]
                    * orientation_flip[prev_axis]
                    >= 0
                ):
                    tick.anchors = ('left', 'center')
                else:
                    tick.anchors = ('right', 'center')

                # TODO: the above flips only left/right anchors. We need to do the same for up/down

        # rotate grids onto the right axes
        for axis in range(3):
            self.grids[axis].transform.rotate(angle=120 * axis, axis=(1, 1, 1))

        self._last_up_direction = up_direction
        self._last_view_is_flipped = view_is_flipped
        self._last_orientation_flip = orientation_flip

    def set_ticks(
        self,
        show_ticks: bool,
        n_ticks: int,
        ranges: list[RangeTuple],
        force: bool = False,
    ) -> None:
        if not show_ticks:
            for axis_ticks in self.tick_labels.values():
                for tick in axis_ticks:
                    tick.visible = False
            return

        # generate tick positions with round values
        tick_positions = np.array(
            [
                compute_nice_ticks(r.start, r.stop, target_ticks=n_ticks)
                for r in ranges
            ]
        )

        if np.array_equal(tick_positions, self._last_ticks) and not force:
            return

        self._last_ticks = tick_positions

        ndim = len(ranges)
        for axis in range(ndim):
            next_axis = (axis + 1) % ndim
            tick_visuals = self.tick_labels[axis]
            new_tick_values = tick_positions[axis]
            for i, val in enumerate(new_tick_values):
                if i >= len(tick_visuals):
                    # more ticks than before, make a new one
                    tick = Text(
                        font_size=8,
                        font_manager=self.font_manager,
                        face=self.font_family,
                    )
                    tick.transform = STTransform()
                    tick_visuals.append(tick)
                else:
                    tick = tick_visuals[i]

                # note the extra newlines and spaces are needed to ensure spacing
                # between the text and the axes regardless of positioning
                tick.text = f'\n  {val:.3g}  \n'
                tick.pos = (val, ranges[next_axis].start, 0)
                tick.color = self.color
                tick.opacity = self._opacity
                tick.parent = self.grids[axis]

            for extra_tick in tick_visuals[len(new_tick_values) :]:
                # disable all extra ones
                extra_tick.parent = None
