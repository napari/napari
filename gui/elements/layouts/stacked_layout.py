from .base_layout import BaseLayout


class StackedLayout(BaseLayout):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.tracked_containers = []

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

        target_size = leading_container.image.shape
        # TODO: account for this when 3D

        h = self.horizontal_axis
        v = self.vertical_axis

        offset = 0
        for container in self.tracked_containers:
            container_size = container.image.shape

            scale = container.scale
            # these differ because the actual visual swaps its axes
            # you can see this via `visual.size` vs `visual._data.shape`
            # yes, it's incredibly confusing
            # TODO: create our own transformation system
            scale[v] = target_size[h] / container_size[h]
            scale[h] = target_size[v] / container_size[v]
            container.scale = scale

            translate = container.translate
            translate[h] = 0
            translate[v] = 0
            translate[2] = offset
            container.translate = translate

            offset -= 1

        self._view_range = ((0, target_size[v]),
                            (0, target_size[h]))

    @classmethod
    def from_layout(cls, layout):
        if isinstance(layout, cls):
            return cls

        from .linear_layout import LinearLayout
        if isinstance(layout, LinearLayout):
            obj = cls(layout.viewer)
            obj.tracked_containers = layout.tracked_containers
            obj.update()
            return obj

        return super().from_layout(layout)
