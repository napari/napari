from .base_layout import BaseLayout


class LinearLayout(BaseLayout):
    def __init__(self, viewer, horizontal=True):
        super().__init__(viewer)
        self.tracked_layers = []

        self.horizontal = horizontal

    @property
    def horizontal(self):
        return self._horizontal

    @horizontal.setter
    def horizontal(self, horizontal):
        self._horizontal = horizontal

        if horizontal:
            self.primary_dim = self.horizontal_axis
            self.secondary_dim = self.vertical_axis
        else:
            self.primary_dim = self.vertical_axis
            self.secondary_dim = self.horizontal_axis

        self.update()

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

        target_height = leading_layer.shape[self.primary_dim]

        offset = 0
        for layer in self.tracked_layers:
            layer_height = layer.shape[self.primary_dim]
            scale = layer.scale
            scale_val = target_height / layer_height
            scale[self.primary_dim] = scale_val
            scale[self.secondary_dim] = scale_val
            layer.scale = scale

            translate = layer.translate
            translate[self.primary_dim] = offset
            translate[self.secondary_dim] = 0
            layer.translate = translate

            offset += layer.display_shape[self.horizontal]

        view_range = ((0, target_height), (0, offset))
        if self.horizontal:
            view_range = view_range[::-1]
        self._view_range = view_range

    @classmethod
    def from_layout(cls, layout, horizontal=None):
        if isinstance(layout, cls):
            if horizontal is not None:
                layout.horizontal = horizontal
            return layout

        if horizontal is None:
            horizontal = False

        from .stacked_layout import StackedLayout
        if isinstance(layout, StackedLayout):
            obj = cls(layout.viewer, horizontal=horizontal)
            obj.tracked_layers = layout.tracked_layers
            obj.update()
            return obj

        return super().from_layout(layout)


class HMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, LinearLayout) and instance.horizontal


class HorizontalLayout(LinearLayout, metaclass=HMeta):
    def __new__(cls, viewer):
        if cls is not HorizontalLayout:
            raise TypeError('Cannot be subclassed.')
        return LinearLayout(viewer, horizontal=True)

    @classmethod
    def from_layout(cls, layout):
        return LinearLayout.from_layout(layout, horizontal=True)


class VMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, LinearLayout) and not instance.horizontal


class VerticalLayout(LinearLayout, metaclass=VMeta):
    def __new__(cls, viewer):
        if cls is not VerticalLayout:
            raise TypeError('Cannot be subclassed.')
        return LinearLayout(viewer, horizontal=False)

    @classmethod
    def from_layout(cls, layout):
        return LinearLayout.from_layout(layout, horizontal=False)
