"""An example of calling a threaded function from a magicgui dock_widget.
Note: this example requires python >= 3.9
"""
from concurrent.futures import Future, ThreadPoolExecutor

from magicgui import magic_factory, widgets
from skimage import data
from skimage.feature import blob_log
from typing_extensions import Annotated

import napari
from napari.types import ImageData, LayerDataTuple


@magic_factory
def make_widget(
    image: ImageData,
    min_sigma: Annotated[float, {"min": 0.5, "max": 15, "step": 0.5}] = 5,
    max_sigma: Annotated[float, {"min": 1, "max": 200, "step": 0.5}] = 30,
    num_sigma: Annotated[int, {"min": 1, "max": 20}] = 10,
    threshold: Annotated[float, {"min": 0, "max": 1000, "step": 0.1}] = 0.3,
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

    with ThreadPoolExecutor() as exec:
        return exec.submit(_make_blob)


viewer = napari.Viewer()
viewer.window.add_dock_widget(make_widget(), area="right")
viewer.add_image(data.hubble_deep_field().mean(-1))

napari.run()
