from vispy.scene import MatrixTransform
from vispy.scene.visuals import GridLines, Node


class Grid(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        self.grids = []
        for _ in range(3):
            grid = GridLines(parent=self, border_width=0)
            self.grids.append(grid)
        self.reset_grid_transforms()

    def reset_grid_transforms(self):
        for i in range(3):
            self.grids[i].transform = MatrixTransform()
            self.grids[i].transform.rotate(angle=120 * i, axis=(1, 1, 1))

    def set_gl_state(self, *args, **kwargs):
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    def set_extents(self, bounds):
        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound

    def set_view_directions(self, bounds, directions):
        self.reset_grid_transforms()
        for i in range(3):
            if directions[i] == 1:
                trans = [0, 0, 0]
                trans[i] = bounds[i]
                self.grids[(i + 1) % 3].transform.translate(trans)
