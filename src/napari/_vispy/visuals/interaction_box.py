from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from vispy.scene.visuals import Compound, Line

from napari._vispy.visuals.markers import Markers
from napari.layers.utils.interaction_box import (
    generate_interaction_box_vertices,
)


class InteractionBox(Compound):
    """Vispy element for displaying an interaction box.

    Visualizes a rectangle with handles at the corners and midpoints,
    and a rotation handle above the top center of the rectangle.

    Vertices are generated according to the following scheme:
    (y is actually upside down in the canvas)
        8
        |
    0---4---2    1 = position
    |       |
    5   9   6
    |       |
    1---7---3
    """

    _edges: ClassVar[npt.NDArray[Any]] = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 8],
        ]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._marker_color = (1, 1, 1, 1)
        self._marker_size = 10
        self._highlight_width = 2
        # squares for corners, diamonds for midpoints, disc for rotation handle
        self._marker_symbol = ['square'] * 4 + ['diamond'] * 4 + ['disc']
        self._edge_color = (0, 0, 1, 1)

        super().__init__([Line(), Markers(antialias=0)], *args, **kwargs)

    @property
    def line(self) -> Line:
        """The interaction box line visual component."""
        return self._subvisuals[0]

    @property
    def markers(self) -> Markers:
        """The interaction box markers visual component."""
        return self._subvisuals[1]

    def set_data(
        self,
        top_left: tuple[float, float],
        bot_right: tuple[float, float],
        handles: bool = True,
        selected: int | None = None,
        rotation: bool = True,
    ) -> None:
        """Update the visualized interaction box with new data.

        Parameters
        ----------
        top_left : tuple[float, float]
            The top left corner of the interaction box.
        bot_right : tuple[float, float]
            The bottom right corner of the interaction box.
        handles : bool
            Whether to show the handles of the interaction box.
        selected : int | None
            The index of the selected handle. If None, no handle is selected.
        rotation : bool
            Whether to show the rotation handle. Default is True.
        """
        vertices = generate_interaction_box_vertices(
            top_left, bot_right, handles=handles, rotation=rotation
        )

        edges = self._edges if handles else self._edges[:4]
        edges = edges if rotation else edges[:4]
        markers = self._marker_symbol if rotation else self._marker_symbol[:8]

        self.line.set_data(pos=vertices, connect=edges)

        if handles:
            marker_edges = np.zeros(len(vertices))
            if selected is not None:
                marker_edges[selected] = self._highlight_width

            self.markers.set_data(
                pos=vertices,
                size=self._marker_size,
                face_color=self._marker_color,
                symbol=markers,
                edge_width=marker_edges,
                edge_color=self._edge_color,
            )
        else:
            self.markers.set_data(pos=np.empty((0, 2)))
