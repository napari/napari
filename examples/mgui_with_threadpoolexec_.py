"""
magicgui with threadpoolexec
============================

An example of calling a threaded function from a magicgui ``dock_widget``.

using ``ThreadPoolExecutor``
Note: this example requires python >= 3.9

.. tags:: gui
"""
import sys
from concurrent.futures import Future, ThreadPoolExecutor

from magicgui import magic_factory
from skimage import data
from skimage.feature import blob_log

import napari
from napari.types import ImageData, LayerDataTuple

if sys.version_info < (3, 9):
    print('This example requires python >= 3.9')
    sys.exit(0)

pool = ThreadPoolExecutor()


@magic_factory(
    min_sigma={"min": 0.5, "max": 15, "step": 0.5},
    max_sigma={"min": 1, "max": 200, "step": 0.5},
    num_sigma={"min": 1, "max": 20},
    threshold={"min": 0, "max": 1000, "step": 0.1},
)
def make_widget(
    image: ImageData,
    min_sigma: float = 5,
    max_sigma: float = 30,
    num_sigma: int = 10,
    threshold: float = 0.3,
) -> Future[LayerDataTuple]:

    # long running function
    def _make_blob():
        # skimage.feature may take a while depending on the parameters
        blobs = blob_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )
        data = blobs[:, : image.ndim]
        kwargs = dict(
            size=blobs[:, -1],
            edge_color="red",
            edge_width=2,
            face_color="transparent",
        )
        return (data, kwargs, 'points')

    return pool.submit(_make_blob)


viewer = napari.Viewer()
viewer.window.add_dock_widget(make_widget(), area="right")
viewer.add_image(data.hubble_deep_field().mean(-1))

napari.run()
pool.shutdown(wait=True)
