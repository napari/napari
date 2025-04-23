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
        self.grids = []
        for _ in range(3):
            grid = GridLines(parent=self, border_width=0)
            grid.transform = MatrixTransform()
            self.grids.append(grid)

    def set_gl_state(self, *args, **kwargs):
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    def set_extents(self, displayed, ranges):
        ndisplay = len(displayed)
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

    def set_view_direction(self, ranges, directions):
        # translate everything with the grid on xy plane, we transpose later
        ndisplay = len(ranges)
        translations = []
        for axis in range(ndisplay):
            self.grids[axis].transform.reset()

            view_inverted = directions[axis]
            farthest_bound = ranges[axis][int(view_inverted)]
            translations.append(farthest_bound)
        if ndisplay == 2:
            translations.append(0)

        for axis in range(3):
            prev_axis = (axis - 1) % 3
            self.grids[axis].transform.translate(
                (0, 0, translations[prev_axis])
            )

        # rotate grids onto the right axes
        for axis in range(3):
            self.grids[axis].transform.rotate(angle=120 * axis, axis=(1, 1, 1))

    def set_ticks(self, show_ticks, tick_spacing, ranges):
        if not show_ticks:
            for axis_ticks in self.tick_labels.values():
                for tick in axis_ticks:
                    tick.visible = False
            return
        if tick_spacing == self._last_spacing:
            return

        ndim = len(ranges)
        for axis in range(ndim):
            self.grids[axis].parent = None
            # clear previous ticks
            for tick in self.tick_labels[axis]:
                tick.parent = None
            self.tick_labels[axis].clear()

            if tick_spacing == 'auto':
                spacing = (
                    ranges[axis].stop - ranges[axis].start
                ) / 5  # TODO: something smarter
            else:
                spacing = tick_spacing[axis]

            for val in np.arange(
                ranges[axis].start, ranges[axis].stop, spacing
            ):
                tick = Text(
                    text=f'{val:.3f}',
                    pos=(val, 0, 0),
                    anchor_x='center',
                    anchor_y='bottom',
                    font_size=8,
                    color='white',
                    parent=self.grids[axis],
                )
                tick.transform = MatrixTransform()
                self.tick_labels[axis].append(tick)

            self.grids[axis].parent = self
