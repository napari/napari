from vispy.scene import MatrixTransform
from vispy.scene.visuals import GridLines, Node


class Grid(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        self.grids = []
        for i in range(3):
            grid = GridLines(parent=self, border_width=0)
            grid.transform = MatrixTransform()
            grid.transform.rotate(angle=120 * i, axis=(1, 1, 1))
            self.grids.append(grid)

    def set_gl_state(self, *args, **kwargs):
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    def set_extents(self, bounds):
        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound
