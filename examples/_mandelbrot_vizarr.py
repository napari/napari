import heapq
import logging
import sys
import threading
from typing import Tuple, Union

import dask.array as da
import numpy as np
import toolz as tz
import zarr
from numba import njit
from numcodecs import Blosc
from psygnal import debounced
from skimage.transform import resize
from superqt import ensure_main_thread
from zarr.storage import init_array, init_group
from zarr.util import json_dumps

import napari
from napari.experimental._progressive_loading import (
    MultiScaleVirtualData, VirtualData, chunk_centers, chunk_priority_2D, chunk_slices,
    get_chunk, interpolated_get_chunk_2D)
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset, openorganelle_mouse_kidney_em)
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.qt.threading import thread_worker
from napari.utils.events import Event

# config.async_loading = True

LOGGER = logging.getLogger("mandelbrot_vizarr")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)

# This global worker is used for fetching data
global worker
worker = None


def get_and_process_chunk_2D(
    chunk_slice,
    scale,
    virtual_data,
    full_shape,
):
    """Fetch and upscale a chunk

    Parameters
    ----------
    chunk_slice : tuple of slices
        a key corresponding to the chunk to fetch
    scale : int
        scale level, assumes power of 2
    array : arraylike
        the ND array to fetch a chunk from
    full_shape : tuple
        a tuple storing the shape of the highest resolution level

    """
    array = virtual_data.array

    
    # Trigger a fetch of the data
    real_array = interpolated_get_chunk_2D(
        chunk_slice,
        array=np.asarray(array),
    )

    # LOGGER.info(
    #     f"\tyield will be placed at: {(y * 2**scale, x * 2**scale, scale, real_array.shape)} slice: {(chunk_slice[0].start, chunk_slice[0].stop, chunk_slice[0].step)} {(chunk_slice[1].start, chunk_slice[1].stop, chunk_slice[1].step)}"
    # )

    # # Catch slices that will be out of bounds during rendering
    # if (
    #     virtual_data.translate[0] + virtual_data.data_plane.shape[0]
    #     < chunk_slice[0].stop
    # ):
    #     import pdb

    #     pdb.set_trace()

    return (
        tuple(chunk_slice),
        scale,
        real_array,
    )


def should_render_scale(scale, viewer):
    layer_name = get_layer_name_for_scale(scale)
    layer = viewer.layers[layer_name]
    layer_shape = layer.data.shape
    layer_scale = layer.scale

    # TODO this dist calculation assumes orientation
    # dist = sum([(t - c)**2 for t, c in zip(layer.translate, viewer.camera.center[1:])]) ** 0.5

    # pixel_size = 2 * np.tan(viewer.camera.angles[-1] / 2) * dist / max(layer_scale)
    pixel_size = viewer.camera.zoom * max(layer_scale)

    # TODO max pixel_size chosen by eyeballing
    # return (pixel_size > 0.25) and (pixel_size < 5)
    return (pixel_size >= 0.5) and (pixel_size <= 4)


@thread_worker
def render_sequence(
    corner_pixels, num_threads=1, visible_scales=[], data=None
):
    """A generator that yields multiscale chunk tuples from low to high resolution.

    Parameters
    ----------
    corner_pixels : tuple
        ND coordinates of the topleft bottomright coordinates of the
        current view
    full_shape : tuple
        shape of highest resolution array
    num_threads : int
        number of threads for multithreaded fetching
    max_scale : int
        this is used to constrain the number of scales that are rendered
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(
        f"render_sequence: inside with corner pixels {corner_pixels} with max_scale {visible_scales} on thread {threading.get_ident()}"
    )

    full_shape = data.arrays[0].shape

    for scale in reversed(range(len(data.arrays))):
        if visible_scales[scale]:
            # This is the real array
            # array = data.arrays[scale]

            array = data._data[scale]

            # these_slices = chunk_slices(array, ndim=self.d)
            # chunk_keys = data._chunk_slices[scale]
            data_interval = corner_pixels / (2 ** scale)
            chunk_keys = chunk_slices(array, ndim=2, interval=data_interval)
            
            LOGGER.info("render_sequence: computing priority")
            chunk_queue = chunk_priority_2D(chunk_keys, corner_pixels, scale)

            LOGGER.info(
                f"render_sequence: {scale}, {array.shape} fetching {len(chunk_queue)} chunks"
            )

            # Fetch all chunks in priority order
            while chunk_queue:
                priority, chunk_slice = heapq.heappop(chunk_queue)
                yield get_and_process_chunk_2D(
                    chunk_slice,
                    scale,
                    array,
                    full_shape,
                )
                
            LOGGER.info(
                f"render_sequence: done fetching {scale}"
            )


def get_layer_name_for_scale(scale):
    return f"scale_{scale}"


@tz.curry
def dims_update_handler(invar, data=None):
    """Start a new render sequence with the current viewer state

    Parameters
    ----------
    invar : Event or viewer
        either an event or a viewer
    full_shape : tuple
        a tuple representing the shape of the highest resolution array

    """
    global worker, viewer

    LOGGER.info("dims_update_handler")

    # This function can be triggered 2 different ways, one way gives us an Event
    if type(invar) is not Event:
        viewer = invar

    # TODO global worker usage is not viable for real implementation
    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.await_workers()
        # worker.await_workers(msecs=10000)

    # Find the corners of visible data in the highest resolution
    corner_pixels = viewer.layers[get_layer_name_for_scale(0)].corner_pixels

    top_left = np.max((corner_pixels,), axis=0)[0, :]
    bottom_right = np.min((corner_pixels,), axis=0)[1, :]

    # TODO we could add padding around top_left and bottom_right to account for future camera movement

    # Interval must be nonnegative
    if not np.all(top_left <= bottom_right):
        import pdb

        pdb.set_trace()

    # TODO Image.corner_pixels behaves oddly maybe b/c VirtualData
    # if bottom_right.shape[0] > 2:
    #     bottom_right[0] = canvas_corners[1, 0]

    corners = np.array([top_left, bottom_right], dtype=np.uint64)

    corners = viewer.layers[get_layer_name_for_scale(0)].corner_pixels

    LOGGER.info(
        f"dims_update_handler: start render_sequence {corners} on layer {get_layer_name_for_scale(0)}"
    )

    # Find the visible scales
    visible_scales = [False] * len(data.arrays)

    for scale in range(len(data.arrays)):
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]
        layer_shape = layer.data.shape
        layer_scale = layer.scale

        layer.metadata["translated"] = False

        # Reenable visibility of layer
        visible_scales[scale] = should_render_scale(scale, viewer)

        layer.visible = visible_scales[scale]
        layer.opacity = 0.9

        scaled_shape = [sh * sc for sh, sc in zip(layer_shape, layer_scale)]

        # TODO this dist calculation assumes orientation
        # dist = sum([(t - c)**2 for t, c in zip(layer.translate, viewer.camera.center[1:])]) ** 0.5

        # pixel_size = 2 * np.tan(viewer.camera.angles[-1] / 2) * dist / max(layer_scale)
        pixel_size = viewer.camera.zoom * max(layer_scale)

        LOGGER.info(
            f"scale {scale} name {layer_name}\twith pixel_size {pixel_size}\ttranslate {layer.data.translate}"
        )

    # TODO toggle visibility of layers

    # TODO ensure that visible_scales has at least 1 visible scale

    # Update the MultiScaleVirtualData memory backing
    data.set_interval(top_left, bottom_right, visible_scales=visible_scales)

    # Start a new multiscale render
    worker = render_sequence(
        corners,
        data=data,
        visible_scales=visible_scales,
    )

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        # TODO bad layer access
        chunk_slice, scale, chunk = coord
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]
        image = viewer.window.qt_viewer.layer_to_visual[
            layer
        ]._layer_node.get_node(2)
        texture = image._texture
        # chunk_size = chunk.shape
        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: {(scale, (chunk_slice[0].start, chunk_slice[0].stop), (chunk_slice[1].start, chunk_slice[1].stop))} in layer {scale} with shape {layer.data.shape} and dataplane shape {layer.data.data_plane.shape} sum {chunk.sum()}"
        )
        # TODO hard coded scale factor
        if not layer.metadata["translated"]:
            layer.translate = np.array(layer.data.translate) * 2**scale

            # Toggle visibility of lower res layer
            if layer.metadata["prev_layer"]:
                # We want to keep prev_layer visible because current layer is loading, but hide others
                if layer.metadata["prev_layer"].metadata["prev_layer"]:
                    layer.metadata["prev_layer"].metadata["prev_layer"].visible = False
                #layer.metadata["prev_layer"].opacity = 0.5
            #     layer.metadata["prev_layer"].visible = False
            layer.metadata["translated"] = True

        LOGGER.info(f"starting set_offset in thread {threading.get_ident()}")
        # TODO check out set_offset and refresh are blocking
        layer.data.set_offset(chunk_slice, chunk)
        # layer.data[chunk_slice] = chunk
        LOGGER.info("done with set_offset of chunk")
        # Refresh is too slow
        # layer.refresh()
        texture.set_data(layer.data.data_plane)
        image.update()

    worker.yielded.connect(on_yield)

    worker.start()


def add_progressive_loading_image(img, viewer=None):
    multiscale_data = MultiScaleVirtualData(img)

    LOGGER.info(f"MultiscaleData {multiscale_data.shape}")

    # Get initial extent for rendering
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.copy()
    canvas_corners[canvas_corners < 0] = 0
    canvas_corners = canvas_corners.astype(np.int64)

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]

    multiscale_data.set_interval(top_left, bottom_right)

    # TODO sketchy Disable _update_thumbnail
    def temp_update_thumbnail(self):
        self.thumbnail = np.ones((32, 32, 4))

    napari.layers.image.Image._update_thumbnail = temp_update_thumbnail

    # We need to initialize the extent of each VirtualData
    layers = {}
    # Start from back to start because we build a linked list

    for scale, vdata in list(enumerate(multiscale_data._data)):
        # TODO scale is assumed to be powers of 2
        layer = viewer.add_image(
            vdata,
            contrast_limits=[0, 255],
            name=get_layer_name_for_scale(scale),
            scale=multiscale_data._scale_factors[scale],
            colormap='PiYG',
        )
        layers[scale] = layer
        layer.metadata["translated"] = False

    # Linked list of layers for visibility control
    for scale in reversed(range(len(layers))):
        layers[scale].metadata["prev_layer"] = (
            layers[scale + 1]
            if scale < len(multiscale_data._data) - 1
            else None
        )

    # TODO initial zoom should not be hardcoded
    # for mandelbrot scales=8
    # viewer.camera.zoom = 0.001
    # for mandelbrot scales=14
    viewer.camera.zoom = 0.0001
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        np.uint64
    )
    LOGGER.info(f"viewer canvas corners {canvas_corners}")

    # Connect to camera and dims
    for listener in [viewer.camera.events, viewer.dims.events]:
        listener.connect(
            debounced(
                ensure_main_thread(dims_update_handler(data=multiscale_data)),
                timeout=1000,
            )
        )

    # Trigger first render
    dims_update_handler(viewer, data=multiscale_data)


if __name__ == "__main__":
    import yappi

    global viewer
    viewer = napari.Viewer()

    def start_yappi():
        yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        yappi.start()

    # large_image = openorganelle_mouse_kidney_em()
    large_image = mandelbrot_dataset()

    multiscale_img = large_image["arrays"]
    viewer._layer_slicer._force_sync = False

    rendering_mode = "progressive_loading"

    if rendering_mode == "progressive_loading":
        # Make an object that creates/manages all scale nodes
        add_progressive_loading_image(multiscale_img, viewer=viewer)
    else:
        layer = viewer.add_image(multiscale_img)

    def stop_yappi():
        yappi.stop()

        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


def yappi_stats():
    import time

    timestamp = time.time()

    filename = f"/tmp/mandelbrot_vizarr_{timestamp}.prof"

    func_stats = yappi.get_func_stats()
    func_stats.save(filename, type='pstat')
