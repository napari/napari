import numpy as np
from vispy.scene.visuals import Compound, Line, Mesh, Text

from napari.layers.shapes._shapes_utils import triangulate_ellipse
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.translations import trans


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


class Axes(Compound):
    def __init__(self) -> None:
        self._num_segments_arrowhead = 100
        # CMYRGB for 6 axes data in x, y, z, ... ordering
        self._default_color = [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]

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
            v, f = make_arrow_head(self._num_segments_arrowhead, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices2D = vertices
        self._default_arrow_faces2D = faces.astype(int)

        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        for axis in range(3):
            v, f = make_arrow_head(self._num_segments_arrowhead, axis)
            faces = np.concatenate([faces, f + len(vertices)], axis=0)
            vertices = np.concatenate([vertices, v], axis=0)
        self._default_arrow_vertices3D = vertices
        self._default_arrow_faces3D = faces.astype(int)

        super().__init__(
            [
                Line(connect='segments', method='gl', width=3),
                Mesh(),
                Text(
                    text='1',
                    font_size=10,
                    anchor_x='center',
                    anchor_y='center',
                ),
            ]
        )

    @property
    def line(self):
        return self._subvisuals[0]

    @property
    def mesh(self):
        return self._subvisuals[1]

    @property
    def text(self):
        return self._subvisuals[2]

    def set_data(self, axes, reversed_axes, colored, bg_color, dashed, arrows):
        ndisplay = len(axes)

        # Determine colors of axes based on reverse position
        if colored:
            axes_colors = [
                self._default_color[ra % len(self._default_color)]
                for ra in reversed_axes
            ]
        else:
            # the reason for using the `as_hex` here is to avoid
            # `UserWarning` which is emitted when RGB values are above 1
            bg_color = transform_color(bg_color.as_hex())[0]
            color = np.subtract(1, bg_color)
            color[-1] = bg_color[-1]
            axes_colors = [color] * ndisplay

        # Determine data based on number of displayed dimensions and
        # axes visualization parameters
        if dashed and ndisplay == 2:
            data = self._dashed_line_data2D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif dashed and ndisplay == 3:
            data = self._dashed_line_data3D
            color = color_dashed_lines(axes_colors)
            text_data = self._line_data3D[1::2]
        elif not dashed and ndisplay == 2:
            data = self._line_data2D
            color = color_lines(axes_colors)
            text_data = self._line_data2D[1::2]
        elif not dashed and ndisplay == 3:
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

        if arrows and ndisplay == 2:
            arrow_vertices = self._default_arrow_vertices2D
            arrow_faces = self._default_arrow_faces2D
            arrow_color = color_arrowheads(
                axes_colors, self._num_segments_arrowhead
            )
        elif arrows and ndisplay == 3:
            arrow_vertices = self._default_arrow_vertices3D
            arrow_faces = self._default_arrow_faces3D
            arrow_color = color_arrowheads(
                axes_colors, self._num_segments_arrowhead
            )
        else:
            arrow_vertices = np.zeros((3, 3))
            arrow_faces = np.array([[0, 1, 2]])
            arrow_color = [[0, 0, 0, 0]]

        self.line.set_data(data, color)
        self.mesh.set_data(
            vertices=arrow_vertices,
            faces=arrow_faces,
            face_colors=arrow_color,
        )

        self.text.color = axes_colors
        self.text.pos = text_data + self._text_offsets
