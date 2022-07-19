from itertools import product

import numpy as np
from vispy.scene.visuals import Compound, Line

from ..visuals.markers import Markers


class BoundingBox(Compound):
    def __init__(self, *args, **kwargs):
        super().__init__([Line(), Markers(antialias=0)], *args, **kwargs)

    def set_bounds(self, node, ndim=2):
        """
        Takes another node to generate its bounding box.
        """

        if ndim not in (2, 3):
            raise ValueError(
                f'Can only compute 2 or 3-dimensional bounds, not {ndim}'
            )

        node._bounds_changed()
        bounds = [node.bounds(i, self) for i in range(ndim)]

        if any(b is None for b in bounds):
            return
        print(1)

        vertices = np.array(list(product(*bounds)))
        if ndim == 3:
            vertices = vertices - 0.5

        # vertices are generated according to the following scheme:
        #    5-------7
        #   /|      /|
        #  1-------3 |
        #  | |     | |
        #  | 4-----|-6
        #  |/      |/
        #  0-------2

        edges = np.array(
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

        # only the front face is needed for 2D
        if ndim == 2:
            edges = edges[:4]

        self._subvisuals[0].set_data(
            pos=vertices, connect=edges, color='red', width=2
        )
        self._subvisuals[1].set_data(
            pos=vertices, face_color='red', edge_width=1, edge_color='black'
        )
