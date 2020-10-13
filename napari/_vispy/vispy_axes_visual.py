import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh
from vispy.visuals.transforms import STTransform

from ..layers.shapes._shapes_utils import triangulate_ellipse


def make_dashed_line(num_dashes, axis):
    """Make a dashed line.

    Parameters
    ----------
    num_dashes : int
        Number of dashes in the line.
    axis :
        Axis which is dashed.

    Returns
    -------
    np.ndarray
        Dashed line.
    """
    dashes = np.linspace(0, 1, num_dashes * 2)
    dashed_line_ends = np.concatenate(
        [[dashes[2 * i], dashes[2 * i + 1]] for i in range(num_dashes)], axis=0
    )
    dashed_line = np.zeros((2 * num_dashes, 3))
    dashed_line[:, axis] = np.array(dashed_line_ends)
    return dashed_line


def make_arrow_head(num_segments, axis):
    """Make an arrowhead line.

    Parameters
    ----------
    num_segments : int
        Number of segments in the arrowhead.
    axis :
        Arrowhead direction.

    Returns
    -------
    np.ndarray, np.ndarray
        Vertices and faces of the arrowhead.
    """
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.1
    vertices, faces = triangulate_ellipse(corners, num_segments)
    full_vertices = np.zeros((num_segments + 1, 3))
    inds = list(range(3))
    inds.pop(axis)
    full_vertices[:, inds] = vertices
    full_vertices[:, axis] = 0.9
    full_vertices[0, axis] = 1.02
    return full_vertices, faces


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation."""

    def __init__(self, axes, parent=None, order=0):
        self.axes = axes

        self._default_data = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )
        self._default_color = np.array(
            [
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 1],
            ]
        )
        self._dashed_data = np.concatenate(
            [
                [[0, 0, 0], [1, 0, 0]],
                make_dashed_line(4, axis=1),
                make_dashed_line(8, axis=2),
            ],
            axis=0,
        )
        self._dashed_color = np.concatenate(
            [
                [[0, 1, 1, 1]] * 2,
                [[1, 1, 0, 1]] * 4 * 2,
                [[1, 0, 1, 1]] * 8 * 2,
            ],
            axis=0,
        )
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(3):
            v, f = make_arrow_head(100, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices = vertices
        self._default_arrow_faces = faces.astype(int)
        self._default_arrow_color = np.concatenate(
            [
                [[0, 1, 1, 1]] * 100,
                [[1, 1, 0, 1]] * 100,
                [[1, 0, 1, 1]] * 100,
            ],
            axis=0,
        )
        self._target_length = 80
        self.node = Compound(
            [Line(connect='segments', method='gl', width=3), Mesh()],
            parent=parent,
        )
        self.node.transform = STTransform()
        self.node.order = order

        self.axes.events.visible.connect(self._on_visible_change)
        self.axes.events.colored.connect(self._on_data_change)
        self.axes.events.dashed.connect(self._on_data_change)
        self.axes.events.arrows.connect(self._on_data_change)

        self._on_visible_change(None)
        self._on_data_change(None)

        self._scale = 0.1
        self.update_scale(1)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self.axes.visible

    def _on_data_change(self, event):
        """Change style of axes."""
        if self.axes.dashed:
            data = self._dashed_data
        else:
            data = self._default_data

        if not self.axes.colored:
            color = np.subtract(1, self.axes.background_color)[:3]
            arrow_color = [color] * 300
        else:
            if self.axes.dashed:
                color = self._dashed_color
                arrow_color = self._default_arrow_color
            else:
                color = self._default_color
                arrow_color = self._default_arrow_color

        if self.axes.arrows:
            arrow_vertices = self._default_arrow_vertices
            arrow_faces = self._default_arrow_faces
        else:
            arrow_vertices = np.zeros((3, 3))
            arrow_faces = np.array([[0, 1, 2]])
            arrow_color = [[0, 0, 0, 0]]

        self.node._subvisuals[0].set_data(data, color)
        self.node._subvisuals[1].set_data(
            vertices=arrow_vertices, faces=arrow_faces, face_colors=arrow_color
        )

    def update_scale(self, scale):
        """Update axes length based on canvas2world scale.

        Parameters
        ----------
        scale : float
            Scale going from canvas pixels to world coorindates.
        """
        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale

        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        scale = target_canvas_pixels * scale_canvas2world
        # Update axes scale
        self.node.transform.scale = [scale, scale, scale, 1]
