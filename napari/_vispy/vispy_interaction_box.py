from vispy.scene.visuals import Line, Compound, Markers
import numpy as np


class VispyInteractionBox:
    def __init__(self, viewer):

        self._viewer = viewer
        self.node = Compound([Line(), Markers()])
        self.layer = None
        self._viewer.events.active_layer.connect(self._on_active_layer_change)

    @property
    def marker_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[1]

    @property
    def line_node(self):
        """sequence of float: Scale factors."""
        return self.node._subvisuals[0]

    def _on_active_layer_change(self, event):
        if self.layer is not None:
            self.layer._interaction_box.events.points_changed.disconnect(
                self._on_interaction_box_change
            )
        self.layer = event.item
        if self.layer is not None:
            self.layer._interaction_box.events.points_changed.connect(
                self._on_interaction_box_change
            )
        self._on_interaction_box_change()

    def _on_interaction_box_change(self, event=None):
        """Called whenever the interaction box changed.
        """
        if self.layer is None:
            self.marker_node.set_data(np.zeros((1, 2)))
            self.line_node.set_data(np.zeros((1, 2)))
            return

        # Compute the location and properties of the vertices and box that
        # need to get rendered
        (
            vertices,
            face_color,
            edge_color,
            pos,
            width,
        ) = self.layer._interaction_box._compute_vertices_and_box()

        if vertices is None or len(vertices) == 0:
            vertices = np.zeros((1, self.layer.dims.ndisplay))
            size = 0
        else:
            vertices = vertices + 0.5
            size = 10

        self.marker_node.set_data(
            vertices,
            size=size,
            face_color=face_color,
            edge_color=edge_color,
            edge_width=1.5,
            symbol='square',
            scaling=False,
        )

        if pos is None or len(pos) == 0:
            pos = np.zeros((1, self.layer.dims.ndisplay))
            width = 0
        else:
            pos = pos + 0.5

        self.line_node.set_data(pos=pos, color=edge_color, width=width)
