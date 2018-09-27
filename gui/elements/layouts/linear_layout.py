from .base_layout import BaseLayout


class LinearLayout(BaseLayout):
    def __init__(self, viewer, horizontal=True):
        super().__init__(viewer)
        self.tracked_containers = []

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

    def add_container(self, container):
        self.tracked_containers.append(container)
        self.update()

    def remove_container(self, container):
        self.tracked_containers.remove(container)
        self.update()

    def __iter__(self):
        return iter(self.tracked_containers)

    def update(self):
        try:
            leading_container = self.tracked_containers[0]
        except IndexError:
            return

        target_height = leading_container.image.shape[self.primary_dim]

        offset = 0
        for container in self.tracked_containers:
            container_height = container.image.shape[self.primary_dim]
            scale = container.scale
            scale_val = target_height / container_height
            scale[self.primary_dim] = scale_val
            scale[self.secondary_dim] = scale_val
            container.scale = scale

            translate = container.translate
            translate[self.primary_dim] = offset
            translate[self.secondary_dim] = 0
            container.translate = translate

            offset += container.display_shape[self.horizontal]

        view_range = ((0, target_height), (0, offset))
        if self.horizontal:
            view_range = view_range[::-1]
        self.viewer.view.camera.set_range(*view_range)

    @classmethod
    def from_layout(cls, layout):
        if isinstance(layout, cls):
            return layout

        return super().from_layout(layout)


class HorizontalLayout(LinearLayout):
    def __new__(cls, viewer):
        if cls is not HorizontalLayout:
            raise TypeError('Cannot be subclassed.')
        return LinearLayout(viewer, horizontal=True)

    @classmethod
    def from_layout(cls, layout):
        if isinstance(layout, (LinearLayout, VerticalLayout)):
            layout.horizontal = True
            return layout

        return LinearLayout.from_layout(layout)

    def __instancecheck__(cls, instance):
        return isistance(instance, LinearLayout) and instance.horizontal


class VerticalLayout(LinearLayout):
    def __new__(cls, viewer):
        if cls is not VerticalLayout:
            raise TypeError('Cannot be subclassed.')
        return LinearLayout(viewer, horizontal=False)

    @classmethod
    def from_layout(cls, layout):
        if isinstance(layout, (LinearLayout, HorizontalLayout)):
            layout.horizontal = False
            return layout

        return LinearLayout.from_layout(layout)

    def __instancecheck__(cls, instance):
        return isistance(instance, LinearLayout) and not instance.horizontal
