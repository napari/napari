import numpy as np
from vispy.scene.visuals import Compound, Line

from napari._vispy.visuals.markers import Markers
from napari.layers.utils.interaction_box import (
    generate_interaction_box_vertices,
)


class InteractionBox(Compound):
    # vertices are generated according to the following scheme:
    # (y is actually upside down in the canvas)
    #      8
    #      |
    #  0---4---2    1 = position
    #  |       |
    #  5   9   6
    #  |       |
    #  1---7---3
    _edges = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [4, 8],
        ]
    )

    def __init__(self, *args, **kwargs):
        self._marker_color = (1, 1, 1, 1)
        self._marker_size = 10
        self._highlight_width = 2
        # TODO: change to array (square + circle for handle) after #5312
        self._marker_symbol = 'square'
        self._edge_color = (0, 0, 1, 1)

        super().__init__([Line(), Markers(antialias=0)], *args, **kwargs)
        # TODO: vispy 0.12 will allow to set different symbols per marker
        self.markers.symbol = self._marker_symbol

    @property
    def line(self):
        return self._subvisuals[0]

    @property
    def markers(self):
        return self._subvisuals[1]

    def set_data(self, top_left, bot_right, handles=True, selected=None):
        vertices = generate_interaction_box_vertices(
            top_left, bot_right, handles=handles
        )

        edges = self._edges if handles else self._edges[:4]

        self.line.set_data(pos=vertices, connect=edges)

        if handles:
            marker_edges = np.zeros(len(vertices))
            if selected is not None:
                marker_edges[selected] = self._highlight_width

            self.markers.set_data(
                pos=vertices,
                size=self._marker_size,
                face_color=self._marker_color,
                # symbol=self._marker_symbol,
                edge_width=marker_edges,
                edge_color=self._edge_color,
            )
        else:
            self.markers.set_data(pos=np.empty((0, 2)))
