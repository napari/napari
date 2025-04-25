import numpy as np
from vispy.scene import MatrixTransform
from vispy.scene.visuals import GridLines, Node, Text


class GridLines3D(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        self.tick_labels = {0: [], 1: [], 2: []}
        self._last_spacing = None
        self._color = 'white'
        self._scale = (1, 1, 1)
        self.grids = []

        self.reset_grids('white')

    def reset_grids(self, color):
        # color and scale are not exposed on the visual, so this is how we set them...
        self._color = color

        for grid in self.grids:
            grid.parent = None
        self.grids.clear()

        for _ in range(3):
            grid = GridLines(
                parent=self,
                border_width=0,
                color=self._color,
                scale=self._scale,
            )
            grid.transform = MatrixTransform()
            self.grids.append(grid)

    def set_gl_state(self, *args, **kwargs):
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    def set_extents(self, ranges):
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

    def set_view_direction(self, ranges, view_reversed, up_direction):
        # translate everything with the grid on xy plane, we transpose later
        if len(ranges) == 2:
            ranges.append((0, 0))
            view_reversed.append(0)
        far_bounds = []
        for axis in range(3):
            self.grids[axis].transform.reset()
            for tick in self.tick_labels[axis]:
                tick.transform.reset()

            # get the translation necessary to bring the grid to the back
            farthest_bound = ranges[axis][int(view_reversed[axis])]
            far_bounds.append(farthest_bound)

        for axis in range(3):
            prev_axis = (axis - 1) % 3
            next_axis = (axis + 1) % 3
            self.grids[axis].transform.translate((0, 0, far_bounds[prev_axis]))

            for tick in self.tick_labels[axis]:
                # undo shifts caused by grid transform so we're back to the axes
                tick.transform.translate((0, 0, -far_bounds[prev_axis]))

                # shift according to view angle to maximize visibility and have consistent positioning
                # these branches were found by trial and error with the goal to reproduce the
                # tick positioning by plotly (e.g: https://plotly.com/python/3d-scatter-plots/)
                next_axis_shift = ranges[next_axis][1] - ranges[next_axis][0]
                prev_axis_shift = ranges[prev_axis][1] - ranges[prev_axis][0]

                if axis == 0:
                    anchor_flip = -1
                    if not view_reversed[next_axis]:
                        tick.transform.translate((0, next_axis_shift, 0))
                        anchor_flip *= -1
                    if view_reversed[prev_axis]:
                        tick.transform.translate((0, 0, prev_axis_shift))
                        anchor_flip *= -1
                if axis == 1:
                    anchor_flip = 1
                    if view_reversed[next_axis]:
                        tick.transform.translate((0, next_axis_shift, 0))
                        anchor_flip *= -1
                    if not view_reversed[prev_axis]:
                        tick.transform.translate((0, 0, prev_axis_shift))
                        anchor_flip *= -1
                if axis == 2:
                    anchor_flip = -1
                    if not view_reversed[next_axis]:
                        tick.transform.translate((0, 0, prev_axis_shift))
                    if view_reversed[prev_axis]:
                        tick.transform.translate((0, next_axis_shift, 0))

                if up_direction[axis] * anchor_flip >= 0:
                    tick.anchors = ('left', 'center')
                else:
                    tick.anchors = ('right', 'center')

        # rotate grids onto the right axes
        for axis in range(3):
            self.grids[axis].transform.rotate(angle=120 * axis, axis=(1, 1, 1))

    def set_ticks(self, show_ticks, tick_spacing, ranges):
        # TODO: this does not work correctly with axes flipping etc
        if not show_ticks:
            for axis_ticks in self.tick_labels.values():
                for tick in axis_ticks:
                    tick.visible = False
            return

        if tick_spacing == 'auto':
            mag = np.log10([(r.stop - r.start) / 5 for r in ranges])
            dec, exp = np.modf(mag)
            exp = np.floor(exp)
            tick_spacing = 10**exp * 2

        if np.array_equal(tick_spacing, self._last_spacing):
            return

        ndim = len(ranges)
        for axis in range(ndim):
            self.grids[axis].parent = None
            # clear previous ticks
            for tick in self.tick_labels[axis]:
                tick.parent = None
            self.tick_labels[axis].clear()
            next_axis = (axis + 1) % ndim

            # TODO: broken if start stop and inverted!
            tick_positions = np.concatenate(
                [
                    np.arange(
                        np.max([0, int(np.floor(ranges[axis].start))]),
                        ranges[axis].stop,
                        tick_spacing[axis],
                    ),
                    np.arange(
                        np.min([0, int(np.ceil(ranges[axis].stop))]),
                        ranges[axis].start,
                        tick_spacing[axis],
                    ),
                ]
            )
            for val in tick_positions:
                tick = Text(
                    text=f'{val:.3f}',
                    pos=(val, ranges[next_axis].start, 0),
                    font_size=8,
                    color=self._color,
                    parent=self.grids[axis],
                )
                tick.transform = MatrixTransform()
                self.tick_labels[axis].append(tick)

            self.grids[axis].parent = self
