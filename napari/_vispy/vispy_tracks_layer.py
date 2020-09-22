from vispy.scene.visuals import Compound, Line, Text

from ._vispy_tracks_shader import TrackShader
from .vispy_base_layer import VispyBaseLayer

# from napari._vispy.vispy_base_layer import VispyBaseLayer
# from ._track_shader import TrackShader


class VispyTracksLayer(VispyBaseLayer):
    """ VispyTracksLayer

    Custom napari Track layer for visualizing tracks.

    Components:
        - Track lines (vispy.LineVisual)
        - Track IDs (vispy.TextVisual)
        - Graph edges (vispy.LineVisual)

    """

    def __init__(self, layer):
        node = Compound([Line(), Text(), Line()])
        super().__init__(layer, node)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.tail_length.connect(self._on_data_change)
        self.layer.events.display_id.connect(self._on_data_change)
        self.layer.events.display_tail.connect(self._on_data_change)
        self.layer.events.display_graph.connect(self._on_data_change)
        self.layer.events.color_by.connect(self._on_data_change)

        # build and attach the shader to the track
        self.track_shader = TrackShader(
            current_time=self.layer.current_time,
            tail_length=self.layer.tail_length,
            vertex_time=self.layer.track_times,
        )

        self.graph_shader = TrackShader(
            current_time=self.layer.current_time,
            tail_length=self.layer.tail_length,
            vertex_time=self.layer.graph_times,
        )

        node._subvisuals[0].attach(self.track_shader)
        node._subvisuals[2].attach(self.graph_shader)

        # text label properties
        self.node._subvisuals[1].color = 'white'
        self.node._subvisuals[1].font_size = 8

        self._reset_base()
        self._on_data_change()

    def _on_data_change(self, event=None):
        """ update the display

        NOTE(arl): this gets called by the VispyBaseLayer

        """
        # update the shader
        self.track_shader.current_time = self.layer.current_time
        self.track_shader.tail_length = self.layer.tail_length
        self.track_shader.use_fade = self.layer.use_fade

        self.graph_shader.current_time = self.layer.current_time
        self.graph_shader.tail_length = self.layer.tail_length
        self.graph_shader.use_fade = self.layer.use_fade

        # set visibility of subvisuals
        self.node._subvisuals[0].visible = self.layer.display_tail
        self.node._subvisuals[1].visible = self.layer.display_id
        self.node._subvisuals[2].visible = self.layer.display_graph

        # change track line width
        self.node._subvisuals[0].set_data(
            pos=self.layer._view_data,
            width=self.layer.edge_width,
            color=self.layer.track_colors,
            connect=self.layer.track_connex,
        )

        # add text labels if they're visible
        if self.node._subvisuals[1].visible:
            labels_text, labels_pos = self.layer.track_labels
            self.node._subvisuals[1].text = labels_text
            self.node._subvisuals[1].pos = labels_pos

        # add the meta-linkages / graph edges
        self.node._subvisuals[2].set_data(
            pos=self.layer._view_graph,
            width=self.layer.edge_width,
            color='white',
            connect=self.layer.graph_connex,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()

    def _on_color_by(self, event=None):
        """ change the coloring only """
        self.node._subvisuals[0].set_data(color=self.layer.track_colors)
        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_scale_change()
        self._on_translate_change()
