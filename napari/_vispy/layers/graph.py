from vispy import gloo

from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.visuals.graph import GraphVisual


class VispyGraphLayer(VispyPointsLayer):
    _visual = GraphVisual
    node: GraphVisual

    def _on_data_change(self) -> None:
        self._set_graph_edges_data()
        super()._on_data_change()

    def _set_graph_edges_data(self) -> None:
        """Sets the LineVisual (subvisual[4]) with the graph edges data"""
        subvisual = self.node._subvisuals[4]
        edges = self.layer._view_edges_coordinates

        if len(edges) == 0:
            subvisual.visible = False
            return

        subvisual.visible = True
        flat_edges = edges.reshape((-1, edges.shape[-1]))  # (N x 2, D)
        flat_edges = flat_edges[:, ::-1]

        # clearing up buffer, there was a vispy error otherwise
        subvisual._line_visual._pos_vbo = gloo.VertexBuffer()
        subvisual.set_data(
            flat_edges,
            color='white',
            width=1,
        )
