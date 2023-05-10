import logging
import sys
from typing import Tuple, Union
import itertools

import concurrent.futures

import numpy as np
import toolz as tz

from psygnal import debounced
from skimage.transform import resize

from superqt import ensure_main_thread

import napari
from napari.experimental._progressive_loading import (
    get_chunk,
    VirtualData,
    MultiScaleVirtualData,
)
from napari.experimental._progressive_loading_datasets import (
    openorganelle_mouse_kidney_em,
    mandelbrot_dataset,
)
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.qt.threading import thread_worker
from napari.utils.events import Event

# config.async_loading = True

LOGGER = logging.getLogger("interpolated_tiled_rendering")
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

"""
Current differences between this (2D) and are_the_chunks_in_view (3D):
- 2D v 3D
- 2D does not use chunk prioritization
- 2D uses linear interpolation
"""


def interpolated_get_chunk(chunk_slice, array=None):
    """Get a specified slice from an array, with interpolation when necessary.
    Interpolation is linear.
    Out of bounds behavior is zeros outside the shape.

    Parameters
    ----------
    coord : tuple
        a float 3D coordinate into the array like (0.5, 0, 0)
    array : ndarray
        one of the scales from the multiscale image

    Returns
    -------
    real_array : ndarray
        an ndarray of data sliced with chunk_slice
    """
    real_array = get_chunk(chunk_slice, array=array)
    if real_array is None:
        # If we do not need to interpolate
        # TODO this isn't safe enough
        if all([(sl.start % 1 == 0) for sl in chunk_slice]):
            real_array = get_chunk(
                chunk_slice,
                array=array,
            )
        else:
            # Get left and right keys
            # TODO int casting may be dangerous
            lchunk_slice = (
                slice(
                    int(np.floor(chunk_slice[0].start - 1)),
                    int(np.floor(chunk_slice[0].stop - 1)),
                ),
                chunk_slice[1],
                chunk_slice[2],
            )
            rchunk_slice = (
                slice(
                    int(np.ceil(chunk_slice[0].start + 1)),
                    int(np.ceil(chunk_slice[0].stop + 1)),
                ),
                chunk_slice[1],
                chunk_slice[2],
            )

            # Handle out of bounds with zeros
            try:
                lvalue = get_chunk(
                    lchunk_slice,
                    array=array,
                )
            except:
                lvalue = np.zeros([1] + list(array.chunksize[-2:]))
            try:
                rvalue = get_chunk(
                    rchunk_slice,
                    array=array,
                )
            except:
                rvalue = np.zeros([1] + list(array.chunksize[-2:]))

            # Linear weight between left/right, assumes parallel
            w = chunk_slice[0].start - lchunk_slice[0].start

            # TODO hardcoded dtype
            # TODO squeeze is a bad sign
            real_array = (
                ((1 - w) * lvalue + w * rvalue).astype(np.uint16).squeeze()
            )
    return real_array


def chunks_for_scale(corner_pixels, array, scale):
    """Return the keys for all chunks at this scale within the corner_pixels

    Parameters
    ----------
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    array : arraylike
        a ND numpy array with the data for this scale
    scale : int
        the scale level, assuming powers of 2

    """

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code

    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    chunk_size = array.chunksize

    # Get the remaining dims
    otherdims = mins[:-2]

    # Find the extent from the current corner pixels, limit by data shape
    # TODO risky int cast
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(int)
    maxs = np.min(
        (np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)),
        axis=0,
    ).astype(int)

    # We'll just use 1 position in each remaining dim
    mins[:-2] = maxs[:-2] = otherdims

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])

    for coord in itertools.product(
        *([[val] for val in otherdims] + [ys] + [xs])
    ):
        other_coords = [slice(val, (val + 1)) for val in coord[:-2]]
        y_min = coord[-2]
        y_max = y_min + chunk_size[-2]
        y_coords = slice(y_min, y_max)
        x_min = coord[-1]
        x_max = x_min + chunk_size[-2]
        x_coords = slice(x_min, x_max)
        yield tuple(other_coords + [y_coords] + [x_coords])


def get_and_process_chunk(chunk_slice, scale, array, full_shape):
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
    # z, y, x = coord

    full_shape = large_image["arrays"][0].shape

    # Trigger a fetch of the data
    dataset = f"{large_image['dataset']}/s{scale}"
    LOGGER.info("render_sequence: get_chunk")
    real_array = interpolated_get_chunk(
        chunk_slice,
        array=array,
    )

    upscale_factor = [el * 2**scale for el in real_array.shape]

    # Upscale the data to highest resolution
    upscaled = resize(
        real_array,
        upscale_factor,
        preserve_range=True,
    )

    # TODO imposes 3D
    z, y, x = [sl.start for sl in chunk_slice]

    # Use this to overwrite data and then use a colormap to debug where resolution levels go
    # upscaled = np.ones_like(upscaled) * scale

    LOGGER.info(
        f"yielding: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)} sample {upscaled[10:20,10]} with sum {upscaled.sum()}"
    )
    # Return upscaled coordinates, the scale, and chunk
    chunk_size = upscaled.shape

    LOGGER.info(
        f"yield will be placed at: {(z * 2**scale, y * 2**scale, x * 2**scale, scale, upscaled.shape)}"
    )

    upscaled_chunk_size = [0, 0]
    upscaled_chunk_size[0] = min(
        full_shape[-2] - y * 2**scale,
        chunk_size[-2],
    )
    upscaled_chunk_size[1] = min(
        full_shape[-1] - x * 2**scale,
        chunk_size[-1],
    )

    upscaled = upscaled[: upscaled_chunk_size[-2], : upscaled_chunk_size[-1]]

    # TODO This is unclean!
    upscaled_chunk_slice = [None] * 3
    for idx, sl in enumerate(chunk_slice):
        start_coord = sl.start * 2**scale
        stop_coord = sl.stop * 2**scale
        # Check for ragged edges
        if idx > 0:
            stop_coord = start_coord + min(
                stop_coord - start_coord, upscaled_chunk_size[idx - 1]
            )

        upscaled_chunk_slice[idx] = slice(start_coord, stop_coord)

    return (
        tuple(upscaled_chunk_slice),
        scale,
        upscaled,
    )


@thread_worker
def render_sequence_2D(corner_pixels, full_shape, num_threads=1):
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
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(f"render_sequence: inside with corner pixels {corner_pixels}")

    arrays = large_image["arrays"]
    
    # TODO scale is hardcoded here
    for scale in reversed(range(len(arrays))):
        array = arrays[scale]

        chunks_to_fetch = list(chunks_for_scale(corner_pixels, array, scale))

        LOGGER.info(
            f"render_sequence: {scale}, {array.shape} fetching {len(chunks_to_fetch)} chunks"
        )

        if num_threads == 1:
            # Single threaded:
            for chunk_slice in chunks_to_fetch:
                yield get_and_process_chunk(
                    chunk_slice,
                    scale,
                    array,
                    full_shape,
                )

        else:
            # Make a list of num_threads length sublists
            all_job_sets = [
                chunks_to_fetch[i] for i in range(0, len(chunks_to_fetch))
            ]

            def mapper(chunk_slice):
                get_and_process_chunk(
                    chunk_slice,
                    scale,
                    array,
                    full_shape,
                )

            with concurrent.futures.ProcessPoolExecutor() as executor:
                # TODO we should really be yielding async instead of in batches
                # Yield the chunks that are done
                for idx, result in enumerate(
                    executor.map(mapper, all_job_sets)
                ):
                    LOGGER.info(
                        f"jobs done: scale {scale} job_idx {idx} with result {result}"
                    )

                    LOGGER.info(
                        f"scale of {scale} upscaled {result[-2].shape} chunksize {result[-1]} at {result[:3]}"
                    )

                    # Return upscaled coordinates, the scale, and chunk
                    yield result


@tz.curry
def dims_update_handler(invar, data=None, layers={}):
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

    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.quit()

    # Find the corners of visible data in the canvas
    corner_pixels = layers[0].corner_pixels
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        int
    )

    top_left = np.max((corner_pixels, canvas_corners), axis=0)[0, :]
    bottom_right = np.min((corner_pixels, canvas_corners), axis=0)[1, :]

    # TODO Image.corner_pixels behaves oddly maybe b/c VirtualData
    if bottom_right.shape[0] > 2:
        bottom_right[0] = canvas_corners[1, 0]

    corners = np.array([top_left, bottom_right], dtype=int)

    LOGGER.info("dims_update_handler: start render_sequence")

    # Get our multiscale data ready
    data.set_interval(top_left, bottom_right)
    
    # Start a new multiscale render
    full_shape = data.shape
    worker = render_sequence_2D(corners, full_shape)

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        chunk_slice, scale, chunk = coord
        layer = layers[scale]
        # chunk_size = chunk.shape
        import pdb; pdb.set_trace()
        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: {(viewer.dims.current_step[0], chunk_slice[0].start, chunk_slice[1].start)}"
        )
        layer.data[chunk_slice] = chunk
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


def add_multiscale_image(viewer, data, contrast_limits=[0, 255]):
    # Get our multiscale data is ready
    min_coord = [0] * len(data.shape)
    max_coord = [0] * len(data.shape)
    
    data.set_interval(min_coord, max_coord)
    
    # Data goes from highest res -> lowest res
    layers = {}
    for scale, scale_data in enumerate(data._data):
        layers[scale] = viewer.add_image(scale_data, name=f"scale{scale}")
    
    # Connect to camera
    viewer.camera.events.connect(
        debounced(
            ensure_main_thread(
                dims_update_handler(
                    data=data,
                    layers=layers
                )
            ),
            timeout=1000,
        )
    )

    # Connect to dims (to get sliders)
    viewer.dims.events.connect(
        debounced(
            ensure_main_thread(
                dims_update_handler(
                    data=data,
                    layers=layers
                )
            ),
            timeout=1000,
        )
    )

    # Trigger first render
    dims_update_handler(
        viewer,
        data=data,
        layers=layers
    )

    pass
    
if __name__ == "__main__":
    global viewer
    viewer = napari.Viewer()

    # Previous
    # large_image = {
    #     "container": "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5",
    #     "dataset": "em/fibsem-uint16",
    #     "scale_levels": 4,
    #     "chunk_size": (384, 384, 384),
    # }
    # large_image["arrays"] = [
    #     read_xarray(
    #         f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
    #         storage_options={"anon": True},
    #     )
    #     for scale in range(large_image["scale_levels"])
    # ]

    # large_image = openorganelle_mouse_kidney_em()
    large_image = mandelbrot_dataset()

    # TODO at least get this size from the image
    data = MultiScaleVirtualData(large_image["arrays"])

    add_multiscale_image(viewer, data, contrast_limits=[0, 255])
    


