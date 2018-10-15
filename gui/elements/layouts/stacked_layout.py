from .base_layout import BaseLayout


class StackedLayout(BaseLayout):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.tracked_layers = []

    def add_layer(self, layer):
        self.tracked_layers.append(layer)
        self.update()

    def remove_layer(self, layer):
        self.tracked_layers.remove(layer)
        self.update()

    def __iter__(self):
        return iter(self.tracked_layers)

    def update(self):
        try:
            leading_layer = self.tracked_layers[0]
        except IndexError:
            return

        target_size = leading_layer.shape
        # TODO: account for this when 3D

        h = self.horizontal_axis
        v = self.vertical_axis

        offset = 0
        for layer in self.tracked_layers:
            layer.z_index = offset

            offset += 1

        size = self.viewer.max_shape
        self._view_range = ((0, size[v]),
                            (0, size[h]))

    @classmethod
    def from_layout(cls, layout):
        if isinstance(layout, cls):
            return cls

        from .linear_layout import LinearLayout
        if isinstance(layout, LinearLayout):
            obj = cls(layout.viewer)
            obj.tracked_layers = layout.tracked_layers
            obj.update()
            return obj

        return super().from_layout(layout)
