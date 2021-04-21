"""An example of calling a threaded function from a magicgui dock_widget."""
from magicgui import magic_factory, widgets
from skimage import feature
from typing_extensions import Annotated
from concurrent.futures import Future

import napari
from napari.types import ImageData, LayerDataTuple


@magic_factory
def make_widget(
    image: ImageData,
    min_sigma: Annotated[float, {"min": 0.5, "max": 15, "step": 0.5}] = 4,
    max_sigma: Annotated[float, {"min": 1, "max": 200, "step": 0.5}] = 120,
    num_sigma: Annotated[int, {"min": 1, "max": 20}] = 10,
    threshold: Annotated[float, {"min": 0, "max": 1000, "step": 0.1}] = 0.3,
) -> Future[LayerDataTuple]:

    from napari.qt import thread_worker

    future: Future[LayerDataTuple] = Future()
    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_widget.insert(0, pbar)  # add progress bar to the top of widget

    def _on_done(result, self=make_widget):
        future.set_result(result)
        self.remove(pbar)

    # long running function
    @thread_worker
    def _make_blob():
        # skimage.feature may take a while depending on the parameters
        blobs = feature.blob_log(
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

    # start the thread
    worker = _make_blob()
    worker.returned.connect(_on_done)
    worker.start()

    # return Future
    return future


viewer = napari.Viewer()
viewer.window.add_dock_widget(make_widget(), area="right")
viewer.open_sample(
    "scikit-image",
    "binary_blobs",
    blob_size_fraction=0.04,
    volume_fraction=0.04,
)

napari.run()
