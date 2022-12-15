from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.visuals.tracks import TracksVisual


class VispyTracksLayer(VispyBaseLayer):
    """VispyTracksLayer

    Track layer for visualizing tracks.

    """

    def __init__(self, layer):
        node = TracksVisual()
        super().__init__(layer, node)

        self.layer.events.tail_width.connect(self._on_appearance_change)
        self.layer.events.tail_length.connect(self._on_appearance_change)
        self.layer.events.head_length.connect(self._on_appearance_change)
        self.layer.events.display_id.connect(self._on_appearance_change)
        self.layer.events.display_tail.connect(self._on_appearance_change)
        self.layer.events.display_graph.connect(self._on_appearance_change)

        self.layer.events.color_by.connect(self._on_appearance_change)
        self.layer.events.colormap.connect(self._on_appearance_change)

        # these events are fired when changes occur to the tracks or the
        # graph - as the vertex buffer of the shader needs to be updated
        # alongside the actual vertex data
        self.layer.events.rebuild_tracks.connect(self._on_tracks_change)
        self.layer.events.rebuild_graph.connect(self._on_graph_change)

        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        """Update the display."""

        # update the shaders
        self.node.tracks_filter.current_time = self.layer.current_time
        self.node.graph_filter.current_time = self.layer.current_time

        # add text labels if they're visible
        if self.node._subvisuals[1].visible:
            labels_text, labels_pos = self.layer.track_labels
            self.node._subvisuals[1].text = labels_text
            self.node._subvisuals[1].pos = labels_pos

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_appearance_change(self):
        """Change the appearance of the data."""

        # update shader properties related to appearance
        self.node.tracks_filter.use_fade = self.layer.use_fade
        self.node.tracks_filter.tail_length = self.layer.tail_length
        self.node.tracks_filter.head_length = self.layer.head_length
        self.node.graph_filter.use_fade = self.layer.use_fade
        self.node.graph_filter.tail_length = self.layer.tail_length
        self.node.graph_filter.head_length = self.layer.head_length

        # set visibility of subvisuals
        self.node._subvisuals[0].visible = self.layer.display_tail
        self.node._subvisuals[1].visible = self.layer.display_id
        self.node._subvisuals[2].visible = self.layer.display_graph

        # set the width of the track tails
        self.node._subvisuals[0].set_data(
            width=self.layer.tail_width,
            color=self.layer.track_colors,
        )
        self.node._subvisuals[2].set_data(
            width=self.layer.tail_width,
        )

    def _on_tracks_change(self):
        """Update the shader when the track data changes."""

        self.node.tracks_filter.use_fade = self.layer.use_fade
        self.node.tracks_filter.tail_length = self.layer.tail_length
        self.node.tracks_filter.vertex_time = self.layer.track_times

        # change the data to the vispy line visual
        self.node._subvisuals[0].set_data(
            pos=self.layer._view_data,
            connect=self.layer.track_connex,
            width=self.layer.tail_width,
            color=self.layer.track_colors,
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_graph_change(self):
        """Update the shader when the graph data changes."""

        # if the user clears a graph after it has been created, vispy offers
        # no method to clear the data, therefore, we need to set private
        # attributes to None to prevent errors
        if self.layer._view_graph is None:
            self.node._subvisuals[2]._pos = None
            self.node._subvisuals[2]._connect = None
            self.node.update()
            return

        # vertex time buffer must change only if data is updated, otherwise vispy buffers might be of different lengths
        self.node.graph_filter.use_fade = self.layer.use_fade
        self.node.graph_filter.tail_length = self.layer.tail_length
        self.node.graph_filter.vertex_time = self.layer.graph_times

        self.node._subvisuals[2].set_data(
            pos=self.layer._view_graph,
            connect=self.layer.graph_connex,
            width=self.layer.tail_width,
            color='white',
        )

        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def reset(self):
        super().reset()
        self._on_appearance_change()
        self._on_tracks_change()
        self._on_graph_change()
