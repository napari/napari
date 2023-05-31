import time
import heapq
import logging
import sys

import numpy as np
import toolz as tz

from psygnal import debounced
from superqt import ensure_main_thread
from napari import Viewer

import napari
from napari.experimental._progressive_loading import (
    MultiScaleVirtualData, chunk_priority_2D,
    chunk_slices, get_chunk)
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset)
from napari.qt.threading import thread_worker
from napari.utils.events import Event

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    """Fetch a chunk.

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

    real_array = np.asarray(array[chunk_slice]).transpose()

    return (
        tuple(chunk_slice),
        scale,
        real_array,
    )

def should_render_scale(scale, viewer, min_scale, max_scale):    
    layer_name = get_layer_name_for_scale(scale)
    layer = viewer.layers[layer_name]
    layer_shape = layer.data.shape
    layer_scale = layer.scale

    pixel_size = viewer.camera.zoom * max(layer_scale)
    
    if max_scale == 7:
        max_pixel = 5
        min_pixel = 0.25
    else:
        max_pixel = 4
        min_pixel = 0.5
    greater_than_min_pixel = pixel_size > min_pixel
    less_than_max_pixel = pixel_size < max_pixel
    render = (greater_than_min_pixel and less_than_max_pixel)

    if not render:
        if scale == min_scale and pixel_size > max_pixel:
            render = True
        elif scale == max_scale and pixel_size < min_pixel:
            render = True

    return render


@thread_worker
def render_sequence(
    corner_pixels, num_threads=1, visible_scales=[], data=None
):
    """Generator that yields chunk tuples from low to high resolution.

    Parameters
    ----------
    corner_pixels : tuple
        ND coordinates of the topleft bottomright coordinates of the
        current view
    full_shape : tuple
        shape of highest resolution array
    num_threads : int
        number of threads for multithreaded fetching
    visible_scales : list
        this is used to constrain the number of scales that are rendered
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(
        f"render_sequence: inside with corner pixels {corner_pixels} with visible_scales {visible_scales}"
    )

    full_shape = data.arrays[0].shape

    for scale in reversed(range(len(data.arrays))):
        if visible_scales[scale]:
            vdata = data._data[scale]

            data_interval = corner_pixels / (2**scale)
            LOGGER.info(
                f"render_sequence: computing chunk slices for {data_interval}"
            )
            chunk_keys = chunk_slices(vdata, ndim=2, interval=data_interval)

            LOGGER.info("render_sequence: computing priority")
            chunk_queue = chunk_priority_2D(chunk_keys, corner_pixels, scale)

            LOGGER.info(
                f"render_sequence: {scale}, {vdata.shape} fetching {len(chunk_queue)} chunks"
            )

            # Fetch all chunks in priority order
            while chunk_queue:
                priority, chunk_slice = heapq.heappop(chunk_queue)

                # TODO consider 1-2 yields per chunk:
                # - first for the target chunk
                # - second for blanking out the lower resolution (is this too wasteful?)                
                yield tuple(
                    list(
                        get_and_process_chunk_2D(
                            chunk_slice,
                            scale,
                            vdata,
                            full_shape,
                        )
                    )
                    + [len(chunk_queue) == 0]
                )

            LOGGER.info(f"render_sequence: done fetching {scale}")


def get_layer_name_for_scale(scale):
    return f"scale_{scale}"


@tz.curry
def dims_update_handler(invar, data=None):
    """Start a new render sequence with the current viewer state.

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
        # worker.await_workers(msecs=30000)

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
    min_scale = 0
    max_scale = len(data.arrays) - 1
    
    for scale in range(len(data.arrays)):
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]
        layer_shape = layer.data.shape
        layer_scale = layer.scale

        layer.metadata["translated"] = False

        # Reenable visibility of layer
        visible_scales[scale] = should_render_scale(scale, viewer, min_scale, max_scale)
        layer.visible = visible_scales[scale]
        layer.opacity = 0.9

        LOGGER.info(
            f"scale {scale} name {layer_name}\twith translate {layer.data.translate}"
        )

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
        chunk_slice, scale, chunk, is_last_chunk = coord

        start_time = time.time()
        # TODO measure timing within on_yield, find the time consumer

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
                    layer.metadata["prev_layer"].metadata[
                        "prev_layer"
                    ].visible = False
            layer.metadata["translated"] = True

        if is_last_chunk:
            if layer.metadata["prev_layer"]:
                layer.metadata["prev_layer"].visible = False

        layer.data.set_offset(chunk_slice, chunk)

        texture.set_data(layer.data.data_plane)

        image.update()
        LOGGER.info(f"{time.time() - start_time} time : done with image update")

    worker.yielded.connect(on_yield)

    worker.start()


def add_progressive_loading_image(img, viewer=None):
    """Add tiled multiscale image"""
    # initialize multiscale virtual data (generate scale factors, translations, and chunk slices)
    multiscale_data = MultiScaleVirtualData(img)

    if not viewer:
        viewer = Viewer()

    LOGGER.info(f"MultiscaleData {multiscale_data.shape}")

    # Get initial extent for rendering
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.copy()    
    canvas_corners[canvas_corners < 0] = 0  # required to cast from float64 to int64
    canvas_corners = canvas_corners.astype(np.int64)

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]
    LOGGER.debug(f'>>> top left: {top_left}, bottom_right: {bottom_right}')
    # set the extents for each scale in data coordinates
    # take the currently visible canvas extents and apply them to the 
    # individual data scales
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
    # viewer.camera.zoom = 0.00001

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]
    LOGGER.debug(f'>>> top left: {top_left}, bottom_right: {bottom_right}')
    LOGGER.info(f"viewer canvas corners {canvas_corners}")

    # Connect to camera and dims
    for listener in [viewer.camera.events, viewer.dims.events]:
        listener.connect(
            debounced(
                ensure_main_thread(dims_update_handler(data=multiscale_data)),
                timeout=2000,
            )
        )

    # Trigger first render
    dims_update_handler(viewer, data=multiscale_data)

    return viewer

if __name__ == "__main__":
    import yappi

    global viewer
    viewer = napari.Viewer()

    def start_yappi():
        yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        yappi.start()

    # large_image = openorganelle_mouse_kidney_em()
    large_image = mandelbrot_dataset(max_levels=14)

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

    napari.run()

    def yappi_stats():
        import time

        timestamp = time.time()

        filename = f"/tmp/mandelbrot_vizarr_{timestamp}.prof"

        func_stats = yappi.get_func_stats()
        func_stats.save(filename, type='pstat')
