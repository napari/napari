import weakref


class BaseLayout:
    def __init__(self, viewer):
        self.viewer = viewer

    @property
    def viewer(self):
        viewer = self._viewer()
        if viewer is None:
            raise ValueError('Lost reference to viewer '
                             '(was garbage collected).')
        return viewer

    @viewer.setter
    def viewer(self, viewer):
        self._viewer = weakref.ref(viewer)

    @property
    def vertical_axis(self):
        return 1  # x-axis

    @property
    def horizontal_axis(self):
        return 0  # y-axis

    @property
    def view_range(self):
        return self._view_range

    def add_layer(self, layer):
        raise NotImplementedError()

    def remove_layer(self, layer):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    @classmethod
    def from_layout(cls, layout):
        raise NotImplementedError()
