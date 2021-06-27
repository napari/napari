import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text
from vispy.visuals.transforms import STTransform

from ..layers.shapes._shapes_utils import triangulate_ellipse
from ..utils.colormaps.standardize_color import transform_color
from ..utils.theme import get_theme
from ..utils.translations import trans


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
    axis
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


def color_lines(colors):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 2],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 2, [colors[2]] * 2],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


def color_dashed_lines(colors):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 4 * 2],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [[colors[0]] * 2, [colors[1]] * 4 * 2, [colors[2]] * 8 * 2],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


def color_arrowheads(colors, num_segments):
    if len(colors) == 2:
        return np.concatenate(
            [[colors[0]] * num_segments, [colors[1]] * num_segments],
            axis=0,
        )
    elif len(colors) == 3:
        return np.concatenate(
            [
                [colors[0]] * num_segments,
                [colors[1]] * num_segments,
                [colors[2]] * num_segments,
            ],
            axis=0,
        )
    else:
        return ValueError(
            trans._(
                'Either 2 or 3 colors must be provided, got {number}.',
                deferred=True,
                number=len(colors),
            )
        )


class VispyAxesVisual:
    """Axes indicating world coordinate origin and orientation."""

    _NUM_SEGMENTS_ARROWHEAD = 100

    def __init__(self, viewer, parent=None, order=0):
        self._viewer = viewer
        self._scale = 1

        # Target axes length in canvas pixels
        self._target_length = 80
        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
        # Text offset from line end position
        self._text_offsets = 0.1 * np.array([1, 1, 1])

        # note order is x, y, z for VisPy
        self._line_data2D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        )
        self._line_data3D = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )

        # note order is x, y, z for VisPy
        self._dashed_line_data2D = np.concatenate(
            [[[1, 0, 0], [0, 0, 0]], make_dashed_line(4, axis=1)],
            axis=0,
        )
        self._dashed_line_data3D = np.concatenate(
            [
                [[1, 0, 0], [0, 0, 0]],
                make_dashed_line(4, axis=1),
                make_dashed_line(8, axis=2),
            ],
            axis=0,
        )

        # note order is x, y, z for VisPy
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(2):
            v, f = make_arrow_head(self._NUM_SEGMENTS_ARROWHEAD, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices2D = vertices
        self._default_arrow_faces2D = faces.astype(int)

        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(3):
            v, f = make_arrow_head(self._NUM_SEGMENTS_ARROWHEAD, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices3D = vertices
        self._default_arrow_faces3D = faces.astype(int)

        self.node = Compound(
            [Line(connect='segments', method='gl', width=3), Mesh(), Text()],
            parent=parent,
        )
        self.node.transform = STTransform()
        self.node.order = order

        # Add a text node to display axes labels
        self.text_node = self.node._subvisuals[2]
        self.text_node.font_size = 10
        self.text_node.anchors = ('center', 'center')
        self.text_node.text = f'{1}'

        # Note:
        # There are issues on MacOS + GitHub action about destroyed
        # C/C++ object during test if those don't get disconnected.
        def set_none():
            self.node._set_canvas(None)
            self.text_node._set_canvas(None)

        self.node.canvas._backend.destroyed.connect(set_none)
        # End Note

        self._viewer.events.theme.connect(self._on_data_change)
        self._viewer.axes.events.visible.connect(self._on_visible_change)
        self._viewer.axes.events.colored.connect(self._on_data_change)
        self._viewer.axes.events.dashed.connect(self._on_data_change)
        self._viewer.axes.events.labels.connect(self._on_data_change)
        self._viewer.axes.events.arrows.connect(self._on_data_change)
        self._viewer.dims.events.order.connect(self._on_data_change)
        self._viewer.dims.events.range.connect(self._on_data_change)
        self._viewer.dims.events.ndisplay.connect(self._on_data_change)
        self._viewer.dims.events.axis_labels.connect(self._on_data_change)
        self._viewer.camera.events.zoom.connect(self._on_zoom_change)

        self._on_visible_change(None)
        self._on_data_change(None)

    def _on_visible_change(self, event):
        """Change visibiliy of axes."""
        self.node.visible = self._viewer.axes.visible
        self._on_zoom_change(event)
        self._on_data_change(event)

    def _on_data_change(self, event):
        """Change style of axes."""
        if not self._viewer.axes.visible:
            return

        # Determine which axes are displayed
        axes = self._viewer.dims.displayed

        # Actual number of displayed dims
        ndisplay = len(self._viewer.dims.displayed)

        # Determine the labels of those axes
        axes_labels = [self._viewer.dims.axis_labels[a] for a in axes[::-1]]
        # Counting backwards from total number of dimensions
        # determine axes positions. This is done as by default
        # the last NumPy axis corresponds to the first Vispy axis
        reversed_axes = [self._viewer.dims.ndim - 1 - a for a in axes[::-1]]

        # Determine colors of axes based on reverse position
        if self._viewer.axes.colored:
            axes_colors = [
                self._default_color[ra % len(self._default_color)]
                for ra in reversed_axes
            ]
        else:
            background_color = get_theme(self._viewer.theme)['canvas']
            background_color = transform_color(background_color)[0]
            color = np.subtract(1, background_color)
            color[-1] = background_color[-1]
            axes_colors = [color] * ndisplay

        # Determine data based on number of displayed dimensions and
        # axes visualization parameters
        if self._viewer.axes.dashed and ndisplay == 2:
            data = self._dashed_line_data2D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif self._viewer.axes.dashed and ndisplay == 3:
            data = self._dashed_line_data3D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data3D[1::2]
        elif not self._viewer.axes.dashed and ndisplay == 2:
            data = self._line_data2D
            color = color_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif not self._viewer.axes.dashed and ndisplay == 3:
            data = self._line_data3D
            color = color_lines(axes_colors)
            text_data = self._line_data3D[1::2]
        else:
            raise ValueError(
                trans._(
                    'Axes dash status and ndisplay combination not supported',
                    deferred=True,
                )
            )

        if self._viewer.axes.arrows and ndisplay == 2:
            arrow_vertices = self._default_arrow_vertices2D
            arrow_faces = self._default_arrow_faces2D
            arrow_color = color_arrowheads(
                axes_colors, self._NUM_SEGMENTS_ARROWHEAD
            )
        elif self._viewer.axes.arrows and ndisplay == 3:
            arrow_vertices = self._default_arrow_vertices3D
            arrow_faces = self._default_arrow_faces3D
            arrow_color = color_arrowheads(
                axes_colors, self._NUM_SEGMENTS_ARROWHEAD
            )
        else:
            arrow_vertices = np.zeros((3, 3))
            arrow_faces = np.array([[0, 1, 2]])
            arrow_color = [[0, 0, 0, 0]]

        self.node._subvisuals[0].set_data(data, color)
        self.node._subvisuals[1].set_data(
            vertices=arrow_vertices,
            faces=arrow_faces,
            face_colors=arrow_color,
        )

        # Set visibility status of text
        self.text_node.visible = (
            self._viewer.axes.visible and self._viewer.axes.labels
        )
        self.text_node.text = axes_labels
        self.text_node.color = axes_colors
        self.text_node.pos = text_data + self._text_offsets

    def _on_zoom_change(self, event):
        """Update axes length based on zoom scale."""
        if not self._viewer.axes.visible:
            return

        scale = 1 / self._viewer.camera.zoom

        # If scale has not changed, do not redraw
        if abs(np.log10(self._scale) - np.log10(scale)) < 1e-4:
            return
        self._scale = scale
        scale_canvas2world = self._scale
        target_canvas_pixels = self._target_length
        scale = target_canvas_pixels * scale_canvas2world
        # Update axes scale
        self.node.transform.scale = [scale, scale, scale, 1]
