import logging
import sys
from typing import Tuple, Union

import concurrent.futures

import numpy as np
import toolz as tz

from psygnal import debounced
from skimage.transform import resize

from superqt import ensure_main_thread

import napari
from napari.experimental._progressive_loading import (
    ChunkCacheManager, get_chunk, openorganelle_mouse_kidney_em)
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


def interpolated_get_chunk(
    chunk_slice, array=None, container=None, dataset=None, cache_manager=None
):
    """Get a specified slice from an array, with interpolation when necessary.
    Interpolation is linear.
    Out of bounds behavior is zeros outside the shape.

    Parameters
    ----------
    coord : tuple
        a float 3D coordinate into the array like (0.5, 0, 0)
    array : ndarray
        one of the scales from the multiscale image
    container: str
        the zarr container name (this is used to disambiguate the cache)
    dataset: str
        the group in the zarr (this is used to disambiguate the cache)
    chunk_size: tuple
        the size of chunk that you want to fetch

    Returns
    -------
    real_array : ndarray
        an ndarray of data sliced with chunk_slice
    """
    real_array = cache_manager.get(container, dataset, chunk_slice)
    if real_array is None:
        # If we do not need to interpolate
        # TODO this isn't safe enough
        if all([(sl.start % 1 == 0) for sl in chunk_slice]):
            real_array = get_chunk(
                chunk_slice,
                array=array,
                container=container,
                dataset=dataset,
                cache_manager=cache_manager,
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
                    container=container,
                    dataset=dataset,
                    cache_manager=cache_manager,
                )
            except:
                lvalue = np.zeros([1] + list(array.chunksize[-2:]))
            try:
                rvalue = get_chunk(
                    rchunk_slice,
                    array=array,
                    container=container,
                    dataset=dataset,
                    cache_manager=cache_manager,
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

        # Save in cache
        cache_manager.put(container, dataset, chunk_slice, real_array)
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

    # TODO kludge for 3D z-only interpolation
    zval = mins[-3]
    
    # Find the extent from the current corner pixels, limit by data shape
    # TODO risky int cast
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(int)
    maxs = np.min(
        (np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)),
        axis=0,
    ).astype(int)

    mins[-3] = maxs[-3] = zval

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])
    zs = [zval]
    # TODO kludge

    for x in xs:
        for y in ys:
            for z in zs:
                yield (
                    slice(z, (z + 1)),
                    slice(y, (y + chunk_size[-2])),
                    slice(x, (x + chunk_size[-1])),
                )


class VirtualData:
    """VirtualData is used to use a 2D array to represent
    a larger shape. The purpose of this function is to provide
    a 2D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.
    """

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

        self.d = 2

        # TODO: I don't like that this is making a choice of slicing axis
        self.data_plane = np.zeros(self.shape[-1 * self.d :], dtype=self.dtype)

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__getitem__(tuple(key[-1 * self.d :]))
        else:
            return self.data_plane.__getitem__(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__setitem__(
                tuple(key[-1 * self.d :]), value
            )
        else:
            return self.data_plane.__setitem__(key, value)


def get_and_process_chunk(chunk_slice, scale, array, full_shape, cache_manager=None):
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
        container=large_image["container"],
        dataset=dataset,
        cache_manager=cache_manager,
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


def mutable_chunk_fetcher(
    results, idx, chunk_slice, scale, array, full_shape, cache_manager=None
):
    """A support function for fetching chunks in a mutable way. This is used to support multithreading.

    Parameters
    ----------
    results : list
        a list for storing results
    idx : int
        index into results to store this chunk
    coord : tuple
        a key corresponding to the chunk to fetch
    scale : int
        scale level, assumes power of 2
    array : arraylike
        the ND array to fetch a chunk from
    full_shape : tuple
        a tuple storing the shape of the highest resolution level
    """
    results[idx] = get_and_process_chunk(
        chunk_slice, scale, array, full_shape, cache_manager=cache_manager
    )


from concurrent.futures import Executor

@thread_worker
def render_sequence(
    corner_pixels, full_shape, num_threads=1, cache_manager=None
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
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(f"render_sequence: inside with corner pixels {corner_pixels}")

    # TODO scale is hardcoded here
    for scale in reversed(range(4)):
        array = large_image["arrays"][scale]

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
                    cache_manager=cache_manager,
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
                    cache_manager=cache_manager,
                )
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # TODO we should really be yielding async instead of in batches
                # Yield the chunks that are done
                for idx, result in enumerate(executor.map(mapper, all_job_sets)):
                        LOGGER.info(
                            f"jobs done: scale {scale} job_idx {idx} with result {result}"
                        )
                
                        LOGGER.info(
                            f"scale of {scale} upscaled {result[-2].shape} chunksize {result[-1]} at {result[:3]}"
                        )
                
                        # Return upscaled coordinates, the scale, and chunk
                        yield result


@tz.curry
def dims_update_handler(invar, full_shape=(), cache_manager=None):
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
    corner_pixels = viewer.layers[0].corner_pixels
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

    # Start a new multiscale render
    worker = render_sequence(corners, full_shape, cache_manager=cache_manager)

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        layer = viewer.layers[0]
        chunk_slice, scale, chunk = coord
        # chunk_size = chunk.shape
        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: {(viewer.dims.current_step[0], chunk_slice[0].start, chunk_slice[1].start)}"
        )
        layer.data[chunk_slice] = chunk
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


if __name__ == "__main__":
    global viewer
    viewer = napari.Viewer()

    cache_manager = ChunkCacheManager()

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

    large_image = openorganelle_mouse_kidney_em()

    # TODO at least get this size from the image
    empty = VirtualData(np.uint16, large_image["arrays"][0].shape)

    # TODO let's choose a chunk size that matches the axis we'll be looking at

    LOGGER.info(f"canvas {empty.shape} and interpolated")

    layer = viewer.add_image(empty, contrast_limits=[20000, 30000])

    # Connect to camera
    viewer.camera.events.connect(
        debounced(
            ensure_main_thread(
                dims_update_handler(
                    full_shape=large_image["arrays"][0].shape,
                    cache_manager=cache_manager,
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
                    full_shape=large_image["arrays"][0].shape,
                    cache_manager=cache_manager,
                )
            ),
            timeout=1000,
        )
    )

    # Trigger first render
    dims_update_handler(
        viewer,
        full_shape=large_image["arrays"][0].shape,
        cache_manager=cache_manager,
    )
