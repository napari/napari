"""An example of calling a threaded function from a magicgui dock_widget."""
import time
from concurrent.futures import Future

import dask.array as da
from dask.distributed import Client
from magicgui import magicgui

import napari
from napari.types import ImageData


def _slow_function(nz):
    time.sleep(2)
    return da.random.random((512, 512, nz))


@magicgui
def widget(nz: int = 1000) -> Future[ImageData]:
    client = Client()
    return client.submit(_slow_function, nz)


viewer = napari.Viewer()
viewer.window.add_dock_widget(widget, area="right")


napari.run()
