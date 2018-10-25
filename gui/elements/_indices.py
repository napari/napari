import weakref


def slice_to_str(s):
    string = ''
    if s.start is not None:
        string += str(s.start)
    string += ':'
    if s.stop is not None:
        string += str(s.stop)
    if s.step is not None:
        string += f':{s.step}'

    return string


class Indices:
    __slots__ = ('_mut', '_viewer', 'axes')

    def __init__(self, viewer, axes=(0, 1)):
        self._mut = [0] * len(axes)
        self.viewer = viewer
        self.axes = axes

    @property
    def viewer(self):
        return self._viewer()

    @viewer.setter
    def viewer(self, viewer):
        self._viewer = weakref.ref(viewer)

    def __getitem__(self, axis):
        if isinstance(axis, slice):
            s = axis
            l = []

            for axis in range(*s.indices(len(self))):
                assert not isinstance(axis, slice)
                l.append(self[axis])

            return l

        if axis in self.axes:
            return slice(None)
        return self._mut[axis]

    def __contains__(self, dim):
        if dim == slice(None):
            return True
        return dim in self._mut

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._mut)

    def __str__(self):
        l = []

        for dim in self:
            if isinstance(dim, slice):
                l.append(slice_to_str(dim))
            else:
                l.append(str(dim))

        return '[' + ', '.join(l) + ']'

    def __repr__(self):
        return f'Indices({str(self)}) at {hex(id(self))}'

    def copy(self):
        return self._mut.copy()

    def refresh(self):
        max_dims = self.viewer.max_dims
        max_shape = self.viewer.max_shape

        curr_dims = len(self)

        if curr_dims > max_dims:
            self._mut = self._mut[:max_dims]
            dims = curr_dims
        else:
            dims = max_dims
            self._mut.extend([0] * (max_dims - curr_dims))
