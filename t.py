import wrapt

import napari
from napari.qt.threading import thread_worker


class ReadOnly(wrapt.ObjectProxy):
    def __getattr__(self, name):
        return ReadOnly(getattr(self.__wrapped__, name))

    def __setattr__(self, name, value):
        print("Cannot set read only view")

    def __repr__(self):
        return repr(self.__wrapped__)


viewer = napari.Viewer()
roviewer = ReadOnly(viewer)

print(thread_worker)
