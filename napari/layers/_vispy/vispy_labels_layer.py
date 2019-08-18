from vispy.scene.visuals import Image as ImageNode
from .vispy_base_layer import VispyBaseLayer
from ...util.misc import interpolate_coordinates
from ..labels._constants import Mode


class VispyLabelsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = ImageNode(None, method='auto')
        super().__init__(layer, node)

        self.node.cmap = self.layer.colormap[1]
        self.node.clim = [0.0, 1.0]
        self._on_data_change()

    def _on_data_change(self):
        self.node._need_colortransform_update = True
        image = self.layer._raw_to_displayed(self.layer._data_view)
        self.node.set_data(image)
        self.node.update()

    def on_mouse_press(self, event):
        """Called whenever mouse pressed in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.layer.position = self._transform_position(list(event.pos))
        coord, label = self.layer.get_value()

        if self.layer._mode == Mode.PAN_ZOOM:
            # If in pan/zoom mode do nothing
            pass
        elif self.layer._mode == Mode.PICKER:
            self.layer.selected_label = label
        elif self.layer._mode == Mode.PAINT:
            # Start painting with new label
            new_label = self.layer.selected_label
            self.layer.paint(coord, new_label)
            self.layer._last_cursor_coord = coord
            self.layer.status = self.layer.get_message(coord, new_label)
        elif self.layer._mode == Mode.FILL:
            # Fill clicked on region with new label
            old_label = label
            new_label = self.layer.selected_label
            self.layer.fill(coord, old_label, new_label)
            self.layer.status = self.layer.get_message(coord, new_label)
        else:
            raise ValueError("Mode not recongnized")

    def on_mouse_move(self, event):
        """Called whenever mouse moves over canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        if event.pos is None:
            return
        self.layer.position = self._transform_position(list(event.pos))

        coord, label = self.layer.get_value()

        if self.layer._mode == Mode.PAINT and event.is_dragging:
            new_label = self.layer.selected_label
            if self.layer._last_cursor_coord is None:
                interp_coord = [coord]
            else:
                interp_coord = interpolate_coordinates(
                    self.layer._last_cursor_coord, coord, self.layer.brush_size
                )
            with self.layer.events.set_data.blocker():
                for c in interp_coord:
                    self.layer.paint(c, new_label)
            self.layer._set_view_slice()
            self.layer._last_cursor_coord = coord
            label = new_label

        self.layer.status = self.layer.get_message(coord, label)

    def on_mouse_release(self, event):
        """Called whenever mouse released in canvas.

        Parameters
        ----------
        event : Event
            Vispy event
        """
        self.layer._last_cursor_coord = None
