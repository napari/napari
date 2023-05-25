from itertools import product

import numpy as np
from vispy.scene.visuals import Compound, Line

from napari._vispy.visuals.markers import Markers


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

    def __init__(self, *args, **kwargs):
        self._marker_color = (1, 1, 1, 1)
        self._marker_size = 1

        super().__init__([Line(), Markers(antialias=0)], *args, **kwargs)

    @property
    def lines(self):
        return self._subvisuals[0]

    @property
    def markers(self):
        return self._subvisuals[1]

    def set_bounds(self, bounds):
        """Update the bounding box based on a layer's bounds."""
        if any(b is None for b in bounds):
            return

        vertices = np.array(list(product(*bounds)))

        self.lines.set_data(
            pos=vertices, connect=self._edges.copy(), color='red', width=2
        )
        self.lines.visible = True

        self.markers.set_data(
            pos=vertices,
            size=self._marker_size,
            face_color=self._marker_color,
            edge_width=0,
        )
