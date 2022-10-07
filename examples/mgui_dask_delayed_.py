"""
magicgui dask delayed
=====================

An example of calling a threaded function from a magicgui dock_widget.
Note: this example requires python >= 3.9

.. tags:: gui
"""
import time
from concurrent.futures import Future

import dask.array as da
from magicgui import magicgui

import napari
from napari.types import ImageData


def _slow_function(nz):
    time.sleep(2)
    return da.random.random((nz, 512, 512))


if __name__ == '__main__':
    from dask.distributed import Client

    client = Client()

    @magicgui(client={'bind': client})
    def widget(client, nz: int = 1000) -> Future[ImageData]:
        return client.submit(_slow_function, nz)

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(widget, area="right")

if __name__ == '__main__':
    napari.run()
