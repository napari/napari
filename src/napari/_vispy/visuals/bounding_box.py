from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.scene.visuals import Compound, Line

from napari._vispy.visuals.markers import Markers
from napari.utils.color import ColorValue

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vispy.visuals.line import LineVisual


class BoundingBox(Compound):
    # vertices are generated according to the following scheme:
    #    5-------7
    #   /|      /|
    #  1-------3 |
    #  | |     | |
    #  | 4-----|-6
    #  |/      |/
    #  0-------2
    _edges = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
        ]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._line_color = ColorValue('red')
        self._line_thickness = 2.0
        self._marker_color = ColorValue((1, 1, 1, 1))
        self._marker_size = 1.0

        super().__init__(
            [Line(antialias=True), Markers(antialias=0)], *args, **kwargs
        )

    @property
    def lines(self) -> LineVisual:
        return self._subvisuals[0]

    @property
    def markers(self) -> Markers:
        return self._subvisuals[1]

    def set_bounds(
        self, bounds: Sequence[Sequence[float] | None] | np.ndarray
    ) -> None:
        """Update the bounding box based on a layer's bounds."""
        if any(b is None for b in bounds):
            return

        vertices = np.array(list(product(*bounds)))

        self.lines.set_data(
            pos=vertices,
            connect=self._edges.copy(),
            color=self._line_color,
            width=self._line_thickness,
        )
        self.lines.visible = True

        self.markers.set_data(
            pos=vertices,
            size=self._marker_size,
            face_color=self._marker_color,
            edge_width=0,
        )
