from itertools import product

import numpy as np
from vispy.scene.visuals import Compound, Line

from ..visuals.markers import Markers


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
        super().__init__(
            [Line(), Line(), Markers(antialias=0)], *args, **kwargs
        )

    @property
    def line2d(self):
        return self._subvisuals[0]

    @property
    def line3d(self):
        return self._subvisuals[1]

    @property
    def markers(self):
        return self._subvisuals[2]

    def _set_bounds_2d(self, vertices):
        # only the front face is needed for 2D
        edges = self._edges[:4]

        self.line2d.set_data(pos=vertices, connect=edges, color='red', width=2)
        self.line2d.visible = True
        self.line3d.visible = False

        self.markers.set_data(
            pos=vertices, face_color='red', edge_width=1, edge_color='black'
        )

    def _set_bounds_3d(self, vertices):
        # pixels in 3D are shifted by half in napari compared to vispy
        # TODO: find exactly where this difference is and write it here
        vertices = vertices - 0.5

        self.line3d.set_data(
            pos=vertices, connect=self._edges.copy(), color='red', width=2
        )
        self.line3d.visible = True
        self.line2d.visible = False

        self.markers.set_data(
            pos=vertices, face_color='red', edge_width=1, edge_color='black'
        )

    def set_bounds(self, node, ndim=2):
        """
        Takes another node to generate its bounding box.
        """

        if ndim not in (2, 3):
            raise ValueError(
                f'Can only compute 2 or 3-dimensional bounds, not {ndim}'
            )

        # vispy caches bounds for visuals for some reason; if they change, we won't know,
        # so we force cache cleaning
        # this is particularly bad for cases like image layer where we initialize the Volume
        # with a 1 pixel texture, which makes bounds permanently stuck at (0, 1)
        node._bounds_changed()
        bounds = [node.bounds(i, self) for i in range(ndim)]
        vertices = np.array(list(product(*bounds)))

        if any(b is None for b in bounds):
            return

        if ndim == 2:
            self._set_bounds_2d(vertices)
        else:
            self._set_bounds_3d(vertices)
