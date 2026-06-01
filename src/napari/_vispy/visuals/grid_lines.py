from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        self.font_size = 8

        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        self.tick_labels: dict[int, list[Text]] = {0: [], 1: [], 2: []}
        self.axis_labels: list[Text] = [
            Text(font_manager=font_manager, face=font_family) for _ in range(3)
        ]
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

        for axis in range(3):
            grid = GridLines(
                parent=self,
                border_width=0,
                color=self.color,
                scale=self.scale,
            )
            grid.transform = MatrixTransform()
            grid.opacity = self._opacity
            self.grids.append(grid)
            self.axis_labels[axis].transform = STTransform()
            self.axis_labels[axis].parent = grid

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
            self.axis_labels[i].pos = (rng0.stop, 0)
            self.axis_labels[i].text = str(i)

        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound

        is_3d = len(bounds) == 3
        self.grids[2].visible = is_3d
        for tick in self.tick_labels[2]:
            tick.visible = is_3d
        self.axis_labels[2].visible = is_3d

        self.update()

    def set_view_direction(
        self,
        ranges: tuple[RangeTuple],
        view_direction: tuple[int, ...],
        orientation_flip: tuple[int, ...],
        zoom: float,
        force: bool = False,
    ) -> None:
        view_is_flipped = tuple(
            d * f >= 0
            for d, f in zip(view_direction, orientation_flip, strict=False)
        )

        if len(ranges) == 2:
            ranges = ranges + ((0, 0),)
            view_is_flipped = view_is_flipped + (False,)

        # translate everything with the grid on xy plane, we transpose later
        far_bounds = []
        for axis in range(3):
            self.grids[axis].transform.reset()
            for tick in self.tick_labels[axis]:
                tick.transform.translate = (0, 0, 0)
            self.axis_labels[axis].transform.translate = (0, 0, 0)

            # get the translation necessary to bring the grid to the back
            # and also the position of the near bound for later
            far_bound_idx = int(view_is_flipped[axis])
            farthest_bound = ranges[axis][far_bound_idx]
            far_bounds.append(farthest_bound)

        # magic numbers, they work ok at a wide range of zooms
        tick_offset = 15 / zoom

        # offset ticks for each axis so they are positioned nicely for readability,
        # putting them all on the outside of the volume (over the background)
        for axis in range(3):
            prev_axis = (axis - 1) % 3
            # move grid to the back of the whole volume (far bound)
            self.grids[axis].transform.translate((0, 0, far_bounds[prev_axis]))

            # shift according to view angle to maximize visibility and have consistent positioning.
            # These branches were found by trial and error with the goal to reproduce the
            # tick positioning by plotly (e.g: https://plotly.com/python/3d-scatter-plots/)

            for tick in self.tick_labels[axis]:
                self._translate_text_based_on_camera(
                    tick,
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
        self, text, axis, far_bounds, ranges, offset, view_is_flipped
    ):
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
                # special case one axis so it's visually nicer (all axes are on the "outside")
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
        force: bool = False,
    ) -> None:
        if not show_ticks:
            for axis_ticks in self.tick_labels.values():
                for tick in axis_ticks:
                    tick.visible = False
            return

        # generate tick positions with round values
        tick_positions = [
            compute_nice_ticks(r.start, r.stop, target_ticks=n_ticks)
            for r in ranges
        ]

        ndim = len(ranges)
        for axis in range(ndim):
            next_axis = (axis + 1) % ndim
            tick_visuals = self.tick_labels[axis]
            new_tick_values = tick_positions[axis]
            for i, val in enumerate(new_tick_values):
                if i >= len(tick_visuals):
                    # more ticks than before, make a new one
                    tick = Text(
                        font_size=self.font_size,
                        font_manager=self.font_manager,
                        face=self.font_family,
                    )
                    tick.transform = STTransform()
                    tick_visuals.append(tick)
                else:
                    tick = tick_visuals[i]

                tick.text = f'{val:.3g}'
                tick.pos = (val, ranges[next_axis].start, 0)
                tick.color = self.color
                tick.opacity = self._opacity
                tick.parent = self.grids[axis]

            for extra_tick in tick_visuals[len(new_tick_values) :]:
                # disable all extra ones
                extra_tick.parent = None

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
