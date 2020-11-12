import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text
from vispy.visuals.transforms import STTransform

from ..layers.shapes._shapes_utils import triangulate_ellipse


def make_dashed_line(num_dashes, axis):
    """Make a dashed line.

    Parameters
    ----------
    num_dashes : int
        Number of dashes in the line.
    axis : int
        Axis which is dashed.

    Returns
    -------
    np.ndarray
        Dashed line, of shape (num_dashes, 3) with zeros in
        the non dashed axes and line segments in the dashed
        axis.
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

    _NUM_SEGMENTS_ARROWHEAD = 100

    def __init__(self, axes, camera, dims, parent=None, order=0):
        self._axes = axes
        self._dims = dims
        self._camera = camera
        self._scale = 1

        # note order is z, y, x
        self._default_color = [[0, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1]]
        self._default_data = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]]
        )
        self._text_offsets = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self._default_line_color = np.concatenate(
            [
                [self._default_color[0]] * 2,
                [self._default_color[1]] * 2,
                [self._default_color[2]] * 2,
            ],
            axis=0,
        )
        # note order is z, y, x
        self._dashed_data = np.concatenate(
            [
                [[0, 0, 0], [0, 0, 1]],
                make_dashed_line(4, axis=1),
                make_dashed_line(8, axis=0),
            ],
            axis=0,
        )
        self._dashed_color = np.concatenate(
            [
                [self._default_color[0]] * 2,
                [self._default_color[1]] * 4 * 2,
                [self._default_color[2]] * 8 * 2,
            ],
            axis=0,
        )
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        # note order is z, y, x
        for axis in range(3):
            v, f = make_arrow_head(self._NUM_SEGMENTS_ARROWHEAD, 2 - axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices = vertices
        self._default_arrow_faces = faces.astype(int)
        self._default_arrow_color = np.concatenate(
            [
                [self._default_color[0]] * self._NUM_SEGMENTS_ARROWHEAD,
                [self._default_color[1]] * self._NUM_SEGMENTS_ARROWHEAD,
                [self._default_color[2]] * self._NUM_SEGMENTS_ARROWHEAD,
            ],
            axis=0,
        )
        self._target_length = 80
        self.node = Compound(
            [Line(connect='segments', method='gl', width=3), Mesh(), Text()],
            parent=parent,
        )
        self.node.transform = STTransform()
        self.node.order = order

        # Add a text node to display axes labels
        self.text_node = self.node._subvisuals[2]
        self.text_node.pos = (
            self._default_data[1::2] + 0.1 * self._text_offsets
        )
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        self._axes.events.visible.connect(self._on_visible_change)
        self._axes.events.colored.connect(self._on_data_change)
        self._axes.events.dashed.connect(self._on_data_change)
        self._axes.events.arrows.connect(self._on_data_change)
        self._dims.events.order.connect(self._on_data_change)
        self._camera.events.zoom.connect(self._on_zoom_change)
        self._dims.events.axis_labels.connect(self._on_data_change)

        self._on_visible_change(None)
        self._on_data_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self._axes.visible
        self.text_node.visible = self._axes.visible
        self._on_zoom_change(None)

    def _on_data_change(self, event):
        """Change style of axes."""
        if self._axes.dashed:
            data = self._dashed_data
        else:
            data = self._default_data

        if not self._axes.colored:
            color = np.subtract(1, self._axes.background_color)[:3]
            arrow_color = [color] * self._NUM_SEGMENTS_ARROWHEAD * 3
        else:
            if self._axes.dashed:
                color = self._dashed_color
                arrow_color = self._default_arrow_color
            else:
                color = self._default_line_color
                arrow_color = self._default_arrow_color

        if self._axes.arrows:
            arrow_vertices = self._default_arrow_vertices
            arrow_faces = self._default_arrow_faces
        else:
            arrow_vertices = np.zeros((3, 3))
            arrow_faces = np.array([[0, 1, 2]])
            arrow_color = [[0, 0, 0, 0]]

        # Color based on displayed dimensions
        order = tuple(self._dims.order[-3:])
        if len(order) == 2:
            # If only two dimensions are displayed pad with a third
            # zeroth axis
            order = (0,) + tuple(np.add(order, 1))
        # map any axes > 2 into the (0, 1, 2) range and reverse the
        # order of the axes to account for numpy to vispy ordering
        order = tuple([i % 3 for i in order[::-1]])

        self.node._subvisuals[0].set_data(data[:, order], color)
        self.node._subvisuals[1].set_data(
            vertices=arrow_vertices[:, order],
            faces=arrow_faces,
            face_colors=arrow_color,
        )

        axis_labels = [self._dims.axis_labels[d] for d in self._dims.displayed]
        self.text_node.text = axis_labels
        self.text_node.color = self._default_color
        self.text_node.pos = data[1::2] + 0.1 * self._text_offsets

    def _on_zoom_change(self, event):
        """Update axes length based on zoom scale.
        """
        if not self._axes.visible:
            return

        scale = 1 / self._camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale
        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        scale = target_canvas_pixels * scale_canvas2world
        # Update axes scale
        self.node.transform.scale = [scale, scale, scale, 1]
