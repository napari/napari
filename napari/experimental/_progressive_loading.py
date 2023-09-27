import heapq
import itertools
import logging
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread

from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.qt.threading import thread_worker

LOGGER = logging.getLogger("napari.experimental._progressive_loading")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


def get_chunk(
    chunk_slice,
    array=None,
    dtype=np.uint8,
    num_retry=3,
):
    """Get a specified slice from an array (uses a cache).

    Parameters
    ----------
    chunk_slice : tuple
        a slice in array space
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
    real_array = None
    retry = 0

    start_time = time.time()

    while real_array is None and retry < num_retry:
        real_array = np.asarray(array[chunk_slice]).transpose()

        retry += 1

    LOGGER.info(f"get_chunk (end) : {(time.time() - start_time)}")

    return real_array


def visual_depth(points, camera):
    """Compute visual depth from camera position to a(n array of) point(s).

    Parameters
    ----------
    points: (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera: napari.components.Camera
        A camera model specifying a view direction and a center or focus point.

    Returns
    -------
    projected_length : (N,) array of float
        Position of the points along the view vector of the camera. These can
        be negative (in front of the center) or positive (behind the center).
    """
    view_direction = camera.view_direction
    points_relative_to_camera = points - camera.center
    projected_length = points_relative_to_camera @ view_direction
    return projected_length


def distance_from_camera_center_line(points, camera):
    """Compute distance from a point or array of points to camera center line.

    This is the line aligned to the camera view direction and passing through
    the camera's center point, aka camera.position.

    Parameters
    ----------
    points: (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera: napari.components.Camera
        A camera model specifying a view direction and a center or focus point.

    Returns
    -------
    distances : (N,) array of float
        Distances from points to the center line of the camera.
    """
    view_direction = camera.view_direction
    projected_length = visual_depth(points, camera)
    projected = view_direction * np.reshape(projected_length, (-1, 1))
    points_relative_to_camera = (
        points - camera.center
    )  # for performance, don't compute this twice in both functions
    distances = np.linalg.norm(projected - points_relative_to_camera, axis=-1)
    return distances


def chunk_centers(array: da.Array, ndim=3):
    """Make a dictionary mapping chunk centers to chunk slices.

    Note: if array is >3D, then the last 3 dimensions are assumed as ZYX
    and will be used for calculating centers

    Parameters
    ----------
    array: dask Array
        The input array.
    ndim: int
        Dimensions of the array.

    Returns
    -------
    chunk_map : dict {tuple of float: tuple of slices}
        A dictionary mapping chunk centers to chunk slices.
    """
    start_pos = [np.cumsum(sizes) - sizes for sizes in array.chunks]
    middle_pos = [
        np.cumsum(sizes) - (np.array(sizes) / 2) for sizes in array.chunks
    ]
    end_pos = [np.cumsum(sizes) for sizes in array.chunks]
    all_start_pos = list(itertools.product(*start_pos))
    # TODO We impose dimensional ordering for ND
    all_middle_pos = [
        el[-ndim:] for el in list(itertools.product(*middle_pos))
    ]
    all_end_pos = list(itertools.product(*end_pos))
    chunk_slices = []
    for start, end in zip(all_start_pos, all_end_pos):
        chunk_slice = [
            slice(start_i, end_i) for start_i, end_i in zip(start, end)
        ]
        # TODO We impose dimensional ordering for ND
        chunk_slices.append(tuple(chunk_slice[-ndim:]))

    mapping = dict(zip(all_middle_pos, chunk_slices))
    return mapping


def chunk_slices(array: da.Array, interval=None) -> list:
    """Create a list of slice objects for each chunk for each dimension.

    Make a dictionary mapping chunk centers to chunk slices.
    Note: if array is >3D, then the last 3 dimensions are assumed as ZYX
    and will be used for calculating centers. If array is <3D, the third
    dimension is assumed to be None.


    Parameters
    ----------
    array: dask or zarr Array
        The input array, a single scale
    interval: iterable (D, n)
        Range in which to limit chunks

    Returns
    -------
    chunk_slices: list of slice objects
        List of slice objects for each chunk for each dimension
    """
    if isinstance(array, da.Array):
        start_pos = [np.cumsum(sizes) - sizes for sizes in array.chunks]
        end_pos = [np.cumsum(sizes) for sizes in array.chunks]
    else:
        # For zarr
        start_pos = []
        end_pos = []
        for dim in range(len(array.chunks)):
            # TODO: the +1 used on stop_idx is related to searchsorted usage
            start_idx, stop_idx = 0, (array.shape[dim] + 1)
            if interval is not None:
                start_idx = (
                    np.floor(interval[0, dim] / array.chunks[dim])
                    * array.chunks[dim]
                )
                stop_idx = (
                    np.ceil(interval[1, dim] / array.chunks[dim])
                    * array.chunks[dim]
                    + 1
                )
            # Inclusive on the end point
            cumuchunks = list(
                range(int(start_idx), int(stop_idx), array.chunks[dim])
            )
            cumuchunks = np.array(cumuchunks)
            start_pos += [cumuchunks[:-1]]
            end_pos += [cumuchunks[1:]]

    if interval is not None:
        for dim in range(len(start_pos)):
            # Find first index in end_pos that is greater than corner_pixels
            first_idx = np.searchsorted(end_pos[dim], interval[0, dim])
            # Find the last index in start_pos that is less than
            # corner_pixels[1,dim]
            last_idx = np.searchsorted(
                start_pos[dim], interval[1, dim], side='right'
            )

            start_pos[dim] = start_pos[dim][first_idx:last_idx]
            end_pos[dim] = end_pos[dim][first_idx:last_idx]

    chunk_slices: List[List] = [[]] * len(array.chunks)
    for dim in range(len(array.chunks)):
        chunk_slices[dim] = [
            slice(st, end) for st, end in zip(start_pos[dim], end_pos[dim])
        ]

    return chunk_slices


@thread_worker
def render_sequence(
    corner_pixels,
    camera,
    visible_scales=None,
    data=None,
    ndisplay=2,
):
    """Generate multiscale chunk tuples from low to high resolution.

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
        f"render_sequence: inside with corner pixels {corner_pixels} with \
        visible_scales {visible_scales}"
    )

    if not visible_scales:
        visible_scales = []

    # TODO 3D needs to change the view interval (e.g. zoom more at each scale)
    for scale in reversed(range(len(data.arrays))):
        if visible_scales[scale]:
            vdata = data._data[scale]

            data_interval = corner_pixels / data._scale_factors[scale]
            LOGGER.info(
                f"render_sequence: computing chunk slices for {data_interval}"
            )
            chunk_keys = chunk_slices(vdata, interval=data_interval)

            LOGGER.info("render_sequence: computing priority")
            chunk_queue = []
            if ndisplay == 2:
                chunk_queue = chunk_priority_2D(
                    chunk_keys, corner_pixels, data._scale_factors[scale]
                )
            elif ndisplay == 3:
                chunk_queue = chunk_priority_3D(
                    chunk_keys,
                    corner_pixels,
                    data._scale_factors[scale],
                    camera=camera,
                )
            else:
                LOGGER.info(
                    f"render_sequence: {ndisplay} dimensions not supported"
                )
                return

            LOGGER.info(
                f"render_sequence: {scale}, {vdata.shape} fetching \
                {len(chunk_queue)} chunks"
            )

            # Fetch all chunks in priority order
            while chunk_queue:
                priority, chunk_slice = heapq.heappop(chunk_queue)

                # TODO consider 1-2 yields per chunk:
                # - first for the target chunk
                # - second for blanking out the lower resolution
                #   (is this too wasteful?)

                # TODO Transpose needed in 2D mandelbrot
                # real_array = np.asarray(vdata.array[chunk_slice]).transpose()

                real_array = np.asarray(vdata.array[chunk_slice]).transpose()

                chunk_result = (
                    tuple(chunk_slice),
                    scale,
                    data._scale_factors,
                    real_array,
                )

                LOGGER.info(
                    f"render_sequence: yielding chunk {chunk_slice} at scale {scale} which has priority\t{priority}"
                )

                yield tuple(list(chunk_result) + [len(chunk_queue) == 0])

                # TODO blank out lower resolution
                # if lower resolution is visible, send zeros

            LOGGER.info(f"render_sequence: done fetching {scale}")


def chunk_keys_within_interval(chunk_keys, mins, maxs):
    """Return chunk_keys that are within interval.

    Returns a dictionary with a list of slices for each dimension
    """
    # contained_keys is an array with list of slices contained along each
    # dimension
    contained_keys: Dict = defaultdict(list)
    for dim, chunk_slices in enumerate(chunk_keys):
        for sl in chunk_slices:
            below_min = sl.start < mins[dim]
            above_max = sl.stop > maxs[dim]
            # If start and stop are below interval, or
            #    start and stop are above interval: return False
            if (below_min and sl.stop < mins[dim]) or (
                above_max and sl.start > maxs[dim]
            ):
                return []
            else:  # noqa: RET505
                contained_keys[dim] += [sl]

    return contained_keys


def get_layer_name_for_scale(scale):
    """Return the layer name for a given scale."""
    return f"scale_{scale}"


@tz.curry
def dims_update_handler(invar, viewer, data=None, ndisplay=None):
    """Start a new render sequence with the current viewer state.

    Parameters
    ----------
    invar : Event or viewer
        either an event or a viewer
    full_shape : tuple
        a tuple representing the shape of the highest resolution array
    """
    # The root layer corresponds to the highest resolution
    root_layer = viewer.layers[get_layer_name_for_scale(0)]

    worker = None
    if "worker" in root_layer.metadata:
        worker = root_layer.metadata["worker"]

    # TODO global worker usage is not viable for real implementation
    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.await_workers()
        # worker.await_workers(msecs=30000)

    # Find the corners of visible data in the highest resolution
    corner_pixels = root_layer.corner_pixels

    top_left = np.max((corner_pixels,), axis=0)[0, :]
    bottom_right = np.min((corner_pixels,), axis=0)[1, :]

    camera = viewer.camera.copy()

    # TODO Added to skip situations when 3D isnt setup on layer yet??
    if np.any((bottom_right - top_left) == 0):
        return

    # TODO we could add padding around top_left and bottom_right to account
    #      for future camera movement

    # Interval must be nonnegative
    if not np.all(top_left <= bottom_right):
        import pdb

        pdb.set_trace()

    LOGGER.info(
        f"dims_update_handler: start render_sequence {corner_pixels} on {root_layer}"
    )

    # Find the visible scales
    visible_scales = [False] * len(data.arrays)
    min_scale = 0
    max_scale = len(data.arrays) - 1

    ndisplay = ndisplay if ndisplay else viewer.dims.ndisplay

    # Get the scale visibility predicate for the correct ndisplay
    should_render_scale = (
        should_render_scale_2D if ndisplay == 2 else should_render_scale_3D
    )

    for scale in range(len(data.arrays)):
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]

        layer.metadata["translated"] = False

        # Reenable visibility of layer
        visible_scales[scale] = should_render_scale(
            scale, viewer, min_scale, max_scale
        )
        layer.visible = visible_scales[scale]
        layer.opacity = 0.9

        LOGGER.info(
            f"scale {scale} name {layer_name}\twith translate \
            {layer.data.translate}"
        )

    # Update the MultiScaleVirtualData memory backing
    data.set_interval(top_left, bottom_right, visible_scales=visible_scales)

    # Start a new multiscale render
    worker = render_sequence(
        corner_pixels,
        data=data,
        visible_scales=visible_scales,
        ndisplay=ndisplay,
        camera=camera,
    )

    LOGGER.info(
        f"dims_update_handler: started render_sequence with corners {corner_pixels}"
    )

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        # TODO bad layer access
        chunk_slice, scale, scale_factors, chunk, is_last_chunk = coord

        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]

        # TODO this relies on the coincidence that node indices are 2 or 3 for
        #      the image and volume members of an Image layer
        node = viewer.window._qt_viewer.layer_to_visual[
            layer
        ]._layer_node.get_node(viewer.dims.ndisplay)

        texture = node._texture

        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: \
            {(scale, [(sl.start, sl.stop) for sl in chunk_slice])} in layer \
            {scale} with shape {layer.data.shape} and dataplane shape \
            {layer.data.hyperslice.shape} sum {chunk.sum()}"
        )

        if not layer.metadata["translated"]:
            layer.translate = (
                np.array(layer.data.translate) * scale_factors[scale]
            )

            # Toggle visibility of lower res layer
            if layer.metadata["prev_layer"]:
                # We want to keep prev_layer visible because current layer is
                # loading, but hide others
                if layer.metadata["prev_layer"].metadata["prev_layer"]:
                    layer.metadata["prev_layer"].metadata[
                        "prev_layer"
                    ].visible = False
            layer.metadata["translated"] = True

        # If this is the last chunk of the layer, turn off the previous layer
        # TODO if chunks are zero-ed when replaced by higher res data,
        #      then delete this
        if is_last_chunk and layer.metadata["prev_layer"]:
            layer.metadata["prev_layer"].visible = False

        layer.data.set_offset(chunk_slice, chunk)

        texture.set_data(layer.data.hyperslice)

        node.update()

    worker.yielded.connect(on_yield)

    root_layer.metadata["worker"] = worker

    worker.start()


def add_progressive_loading_image(
    img,
    viewer=None,
    contrast_limits=None,
    colormap='PiYG',
    ndisplay=2,
    rendering="attenuated_mip",
    scale=None,
):
    """Add tiled multiscale image."""
    # initialize multiscale virtual data (generate scale factors, translations,
    # and chunk slices)
    if contrast_limits is None:
        contrast_limits = [0, 255]
    multiscale_data = MultiScaleVirtualData(img, ndisplay=ndisplay)

    if not viewer:
        from napari import Viewer

        viewer = Viewer()

    # The scale bar will help this be more dramatic
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"

    viewer.dims.ndim = ndisplay
    # Ensure async slicing is enabled
    viewer._layer_slicer._force_sync = False

    LOGGER.info(f"Adding MultiscaleData with shape: {multiscale_data.shape}")

    # Get initial extent for rendering
    canvas_corners = (
        viewer.window._qt_viewer.canvas._canvas_corners_in_world.copy()
    )
    canvas_corners[
        canvas_corners < 0
    ] = 0  # required to cast from float64 to int64
    canvas_corners = canvas_corners.astype(np.int64)

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]

    # TODO This is required when ndisplay does not match the ndim of the data
    if ndisplay != len(img[0].shape):
        top_left = [viewer.dims.point[-ndisplay]] + top_left.tolist()
        bottom_right = [viewer.dims.point[-ndisplay]] + bottom_right.tolist()

    LOGGER.debug(f'>>> top left: {top_left}, bottom_right: {bottom_right}')
    # set the extents for each scale in data coordinates
    # take the currently visible canvas extents and apply them to the
    # individual data scales
    multiscale_data.set_interval(top_left, bottom_right)

    # TODO yikes!
    import napari

    # TODO sketchy Disable _update_thumbnail
    def temp_update_thumbnail(self):
        self.thumbnail = np.ones((32, 32, 4))

    napari.layers.image.Image._update_thumbnail = temp_update_thumbnail

    # We need to initialize the extent of each VirtualData
    layers = {}
    # Start from back to start because we build a linked list

    viewer.dims.ndim = ndisplay

    if scale is None:
        scale = np.array((1, 1, 1))
    else:
        LOGGER.error("scale other than 1 is currently not supported")
        return None
        # scale = np.asarray(scale)

    for scale_idx, vdata in list(enumerate(multiscale_data._data)):
        layer_scale = scale * multiscale_data._scale_factors[scale_idx]
        layer = viewer.add_image(
            vdata,
            name=get_layer_name_for_scale(scale_idx),
            colormap=colormap,
            scale=layer_scale,
            rendering=rendering,
            contrast_limits=contrast_limits,
        )
        layers[scale_idx] = layer
        layer.metadata["translated"] = False

    # Linked list of layers for visibility control
    for scale_idx in reversed(range(len(layers))):
        layers[scale_idx].metadata["prev_layer"] = (
            layers[scale_idx + 1]
            if scale_idx < len(multiscale_data._data) - 1
            else None
        )

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]
    LOGGER.debug(f'>>> top left: {top_left}, bottom_right: {bottom_right}')
    LOGGER.info(f"viewer canvas corners {canvas_corners}")

    # Connect to camera and dims
    for listener in [viewer.camera.events, viewer.dims.events]:
        listener.connect(
            debounced(
                ensure_main_thread(
                    dims_update_handler(
                        data=multiscale_data,
                        viewer=viewer,
                        ndisplay=ndisplay,
                    )
                ),
                timeout=2000,
            )
        )

    # Trigger first render
    dims_update_handler(
        None, data=multiscale_data, viewer=viewer, ndisplay=ndisplay
    )

    return viewer


# ---------- 2D specific ----------


def chunk_priority_2D(chunk_keys, corner_pixels, scale_factor):
    """Return the keys for all chunks at this scale within the corner_pixels.

    Parameters
    ----------
    chunk_keys : list of list of slices for each dimension
        a list of list of slices for each dimension
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    scale_factor : float
        the scale factor for this scale level

    """
    mins = corner_pixels[0, :] / scale_factor
    maxs = corner_pixels[1, :] / scale_factor

    contained_keys = chunk_keys_within_interval(chunk_keys, mins, maxs)

    priority_map: List = []

    for _idx, chunk_key in enumerate(
        list(
            itertools.product(
                *[contained_keys[k] for k in sorted(contained_keys.keys())]
            )
        )
    ):
        priority = 0
        # TODO filter priority here
        priority = 0 if True else np.inf
        if priority < np.inf:
            heapq.heappush(priority_map, (priority, chunk_key))

    return priority_map


def should_render_scale_2D(scale, viewer, min_scale, max_scale):
    """Test if a scale should be rendered.

    Parameters
    ----------
    scale : int
        a scale level
    viewer : napari.viewer.Viewer
        a napari viewer
    min_scale : int
        the minimum scale level to show
    max_scale : int
        the maximum scale level to show
    """
    layer_name = get_layer_name_for_scale(scale)
    layer = viewer.layers[layer_name]
    layer_scale = layer.scale

    pixel_size = viewer.camera.zoom * max(layer_scale)

    # Define bounds of expected pixel size
    max_pixel = 4
    min_pixel = 0.25

    render = min_pixel < pixel_size < max_pixel

    if not render:
        if scale == min_scale and pixel_size > max_pixel:
            render = True
        elif scale == max_scale and pixel_size < min_pixel:
            render = True

    return render


# ---------- 3D specific ----------


def get_chunk_center(chunk_slice):
    """
    Return the center of chunk_slice.


    chunk_slices is a tuple of slices
    """
    return np.array([(sl.start + sl.stop) * 0.5 for sl in chunk_slice])


def chunk_priority_3D(chunk_keys, corner_pixels, scale_factor, camera):
    """Return the keys for all chunks at this scale within the corner_pixels.

    Parameters
    ----------
    chunk_keys : list of list of slices for each dimension
        a list of list of slices for each dimension
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    scale_factor : float
        the scale factor for this scale level

    """
    mins = corner_pixels[0, :] / scale_factor
    maxs = corner_pixels[1, :] / scale_factor

    contained_keys = chunk_keys_within_interval(chunk_keys, mins, maxs)

    priority_map: List = []

    for _idx, chunk_key in enumerate(
        list(
            itertools.product(
                *[contained_keys[k] for k in sorted(contained_keys.keys())]
            )
        )
    ):
        priority = 0

        chunk_center = get_chunk_center(chunk_key)
        depth = visual_depth(chunk_center, camera)
        center_line_dist = distance_from_camera_center_line(
            chunk_center, camera
        )

        # TODO magic numbers
        priority = (depth + camera.zoom * center_line_dist) * (
            1 + 0.0001 * np.random.rand()
        )

        if priority < np.inf:
            heapq.heappush(priority_map, (priority, chunk_key))

    return priority_map


def should_render_scale_3D(scale, viewer, min_scale, max_scale):
    """Test if a scale should be rendered.

    Parameters
    ----------
    scale : int
        a scale level
    viewer : napari.viewer.Viewer
        a napari viewer
    min_scale : int
        the minimum scale level to show
    max_scale : int
        the maximum scale level to show
    """
    layer_name = get_layer_name_for_scale(scale)
    layer = viewer.layers[layer_name]
    layer_scale = layer.scale

    pixel_size = viewer.camera.zoom * max(layer_scale)

    if max_scale == 7:
        max_pixel = 5
        min_pixel = 0.25
    else:
        max_pixel = 10
        min_pixel = 5
    greater_than_min_pixel = pixel_size > min_pixel
    less_than_max_pixel = pixel_size < max_pixel
    render = greater_than_min_pixel and less_than_max_pixel

    if not render:
        if scale == min_scale and pixel_size > max_pixel:
            render = True
        elif scale == max_scale and pixel_size < min_pixel:
            render = True

    return render


# TODO to be deprecated
def prioritized_chunk_loading_3D(
    depth, distance, zoom, alpha=1.0, visible=None
):
    """Compute a chunk priority based on chunk location relative to camera.

    Lower priority is preferred.

    Parameters
    ----------
    depth : (N,) array of float
        The visual depth of the points.
    distance : (N,) array of float
        The distance from the camera centerline of each point.
    zoom : float
        The camera zoom level. The higher the zoom (magnification), the
        higher the relative importance of the distance from the centerline.
    alpha : float
        Parameter weighing distance from centerline and depth. Higher alpha
        means centerline distance is weighted more heavily.
    visible : (N,) array of bool
        An array that indicates the visibility of each chunk

    Returns
    -------
    priority : (N,) array of float
        The loading priority of each chunk.

    Note: priority values of np.inf should not be displayed
    """
    chunk_load_priority = depth + alpha * zoom * distance
    if visible is not None:
        chunk_load_priority[np.logical_not(visible)] = np.inf
    return chunk_load_priority


@thread_worker
def render_sequence_3D_caller(
    view_slice,
    scale=0,
    camera=None,
    arrays=None,
    chunk_maps=None,
    alpha=0.8,
    scale_factors=None,
    dtype=np.uint16,
    dims=None,
):
    """
    Entry point for recursive function render_sequence.

    See render_sequence for docs.
    """
    if scale_factors is None:
        scale_factors = []
    yield from render_sequence_3D(
        view_slice,
        scale=scale,
        camera=camera,
        arrays=arrays,
        chunk_maps=chunk_maps,
        alpha=alpha,
        scale_factors=scale_factors,
        dtype=dtype,
        dims=dims,
    )


def render_sequence_3D(
    view_slice,
    scale=0,
    camera=None,
    arrays=None,
    chunk_maps=None,
    alpha=0.8,
    scale_factors=None,
    dtype=np.uint16,
    dims=None,
):
    """Add multiscale chunks to a napari viewer for a 3D image layer.

    Note: scale levels are assumed to be 2x factors of each other

    Parameters
    ----------
    view_slice : tuple or list of slices
        A tuple/list of slices defining the region to display
    scale : float
        The scale level to display. 0 is highest resolution
    camera : Camera
        a napari Camera used for prioritizing data loading
        Note: the camera instance should be immutable.
    cache_manager : ChunkCacheManager
        An instance of a ChunkCacheManager for data fetching
    arrays : list
        multiscale arrays to display
    chunk_maps : list
        a list of dictionaries mapping chunk coordinates to chunk
        slices
    container : str
        the name of a zarr container, used for making unique keys in
        cache
    dataset : str
        the name of a zarr dataset, used for making unique keys in
        cache
    alpha : float
        a parameter that tunes the behavior of chunk prioritization
        see prioritized_chunk_loading for more info
    scale_factors : list of tuples
        a list of tuples of scale factors for each array
    dtype : dtype
        dtype of data
    """
    # Get some variables specific to this scale level
    if scale_factors is None:
        scale_factors = []
    min_coord = [st.start for st in view_slice]
    max_coord = [st.stop for st in view_slice]
    array = arrays[scale]
    chunk_map = chunk_maps[scale]
    scale_factor = scale_factors[scale]

    # Points for each chunk, for example, centers
    points = np.array(list(chunk_map.keys()))

    # Mask of whether points are within our interval, this is in array
    # coordinates
    point_mask = np.array(
        [
            np.all(point >= min_coord) and np.all(point <= max_coord)
            for point in points
        ]
    )

    # Rescale points to world for priority calculations
    points_world = points * np.array(scale_factor)

    # Prioritize chunks using world coordinates
    distances = distance_from_camera_center_line(points_world, camera)
    depth = visual_depth(points_world, camera)
    priorities = prioritized_chunk_loading_3D(
        depth, distances, camera.zoom, alpha=alpha, visible=point_mask
    )

    # Select the number of chunks
    # TODO consider using threshold on priorities
    """
    Note:
    switching from recursing on 1 top chunk to N-best introduces extra
    complexity, because the shape of texture allocation needs to
    accommodate projections from all viewpoints around the volume.
    """
    n = 1
    best_priorities = np.argsort(priorities)[:n]

    # Iterate over points/chunks and add corresponding nodes when appropriate
    for idx, point in enumerate(points):
        # TODO: There are 2 strategies here:
        # 1. Render *visible* chunks, or all if we're on the last scale level
        #    Skip the chunk at this resolution because it will be shown in
        #    higher res. This fetches less data.
        # if point_mask[idx] and (idx not in best_priorities or scale == 0):
        # 2. Render all chunks because we know we will zero out this data when
        #    it is loaded at the next resolution level.
        if point_mask[idx]:
            coord = tuple(point)
            chunk_slice = chunk_map[coord]
            offset = [sl.start for sl in chunk_slice]

            # When we get_chunk chunk_slice needs to be in data space, but
            # chunk slices are 3D
            data_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [slice(sl.start, sl.stop) for sl in chunk_slice]
            )

            data = get_chunk(
                data_slice,
                array=array,
                dtype=dtype,
            )

            # Texture slice (needs to be in layer.data dimensions)
            # TODO there is a 3D ordering assumption here
            texture_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [
                    slice(sl.start - offset, sl.stop - offset)
                    for sl, offset in zip(chunk_slice, min_coord)
                ]
            )
            if texture_slice[1].start < 0:
                import pdb

                pdb.set_trace()

            # TODO consider a data class instead of a tuple
            yield (
                np.asarray(data),
                scale,
                offset,
                None,
                chunk_slice,
                texture_slice,
            )

    # TODO make sure that all of low res loads first
    # TODO take this 1 step further and fill all high resolutions with low res

    # recurse on best priorities
    if scale > 0:
        # The next priorities for loading in higher resolution are the best
        # ones
        for priority_idx in best_priorities:
            # Get the coordinates of the chunk for next scale
            priority_coord = tuple(points[priority_idx])
            chunk_slice = chunk_map[priority_coord]

            # Blank out the region that will be filled in by other scales
            zeros_size = list(array.shape[:-3]) + [
                sl.stop - sl.start for sl in chunk_slice
            ]

            zdata = np.zeros(np.array(zeros_size, dtype=dtype), dtype=dtype)

            # TODO there is a 3D ordering assumption here
            texture_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [
                    slice(sl.start - offset, sl.stop - offset)
                    for sl, offset in zip(chunk_slice, min_coord)
                ]
            )

            # Compute the relative scale factor between these layers
            relative_scale_factor = [
                this_scale / next_scale
                for this_scale, next_scale in zip(
                    scale_factors[scale], scale_factors[scale - 1]
                )
            ]

            # now convert the chunk slice to the next scale
            next_chunk_slice = [
                slice(st.start * dim_scale, st.stop * dim_scale)
                for st, dim_scale in zip(chunk_slice, relative_scale_factor)
            ]

            next_min_coord = [st.start for st in next_chunk_slice]
            # TODO this offset is incorrect
            next_world_offset = np.array(next_min_coord) * np.array(
                scale_factors[scale - 1]
            )

            # TODO Note that we need to be blanking out lower res data at the
            #      same time
            # TODO this is when we should move the node from the next
            #      resolution.
            yield (
                np.asarray(zdata),
                scale,
                tuple([sl.start for sl in chunk_slice]),
                next_world_offset,
                chunk_slice,
                texture_slice,
            )

            # Start the next scale level
            yield from render_sequence_3D(
                next_chunk_slice,
                scale=scale - 1,
                camera=camera,
                arrays=arrays,
                chunk_maps=chunk_maps,
                scale_factors=scale_factors,
                dtype=dtype,
                dims=dims,
            )


def interpolated_get_chunk_2D(chunk_slice, array=None):
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
    real_array = None

    start_time = time.time()

    if real_array is None:
        # If we do not need to interpolate
        # TODO this isn't safe enough
        if all((sl.start % 1 == 0) for sl in chunk_slice):
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
            except Exception:
                lvalue = np.zeros([1] + list(array.chunksize[-2:]))
            try:
                rvalue = get_chunk(
                    rchunk_slice,
                    array=array,
                )
            except Exception:
                rvalue = np.zeros([1] + list(array.chunksize[-2:]))

            # Linear weight between left/right, assumes parallel
            w = chunk_slice[0].start - lchunk_slice[0].start

            # TODO squeeze is a bad sign
            real_array = ((1 - w) * lvalue + w * rvalue).squeeze()

    LOGGER.info(f"interpolated_get_chunk_2D : {(time.time() - start_time)}")

    return real_array


class VirtualData:
    """`VirtualData` wraps a particular scale level's array.

    It acts like an array of that size, but only works within the interval
    setup by `set_interval`. Each `VirtualData` uses the scale level's
    coordinates.

    -- `VirtualData` uses a hyperslice to store the currently active interval.
    -- `VirtualData.translate` specifies the offset of the
    `VirtualData.hyperslice`'s origin from `VirtualData.array`'s origin, in
    `VirtualData`'s coordinate system

    VirtualData is used to use a ND array to represent
    a larger shape. The purpose of this function is to provide
    a ND slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last `ndisplay` dimensions
    will be used as the (y, x) values.

    NEW: use a translate to define subregion of image

    Attributes
    ----------
    array: ndarray-like
        nd-array like probably a Dask array or a zarr Array
    dtype: dtype
        dtype of the array
    shape: tuple
        shape of the true data (not the hyperslice)
    ndim: int
        Number of dimensions for this scale
    translate: list[tuple(int)]
        tuple for the translation
    d: int
        Dimension of the chunked slices ??? Hard coded to 2.
    hyperslice: dask.array
        Array of currently visible data for this layer
    _min_coord: list
        List of the minimum coordinates in each dimension
    _max_coord: list
        List of the maximum coordinates in each dimension

    """

    def __init__(self, array, scale, ndisplay=2):
        self.array = array
        self.dtype = array.dtype
        # This shape is the shape of the true data, but not our hyperslice
        self.shape = array.shape
        self.ndim = len(self.shape)

        # translate is in the same units as the highest resolution scale
        self.translate = tuple([0] * len(self.shape))

        self.hyperslice = da.zeros(1)
        self.ndisplay = ndisplay

        self._max_coord = None
        self._min_coord = None
        self.scale = scale  # for debugging

    def set_interval(self, coords):
        """Interval is the data for this scale that is currently visible.

        It is stored in `_min_coord` and `_max_coord` in the coordinates of
        the original array.

        This function takes a slice, converts it to a range (stored as
        `_min_coord` and `_max_coord`), and extracts a subset of the orginal
        data array (stored as `hyperslice`)

        Parameters
        ----------
        coords: tuple(slice(ndim))
            tuple of slices in the same coordinate system as the parent array.
        """
        # store the last interval
        prev_max_coord = self._max_coord
        prev_min_coord = self._min_coord

        # extract the coordinates as a range
        self._max_coord = [sl.stop for sl in coords]
        self._min_coord = [sl.start for sl in coords]

        # Round max and min coord according to self.array.chunks
        # for each dimension, reset the min/max coords, aka interval to be
        # the range of chunk coordinates since we can't partially load a chunk
        for dim in range(len(self._max_coord)):
            if isinstance(self.array, da.Array):
                chunks = self.array.chunks[dim]
                cumuchunks = np.array(chunks).cumsum()
            else:
                # For zarr
                cumuchunks = list(
                    range(
                        self.chunks[dim],
                        self.array.shape[dim],
                        self.chunks[dim],
                    )
                )
                # Add last element
                cumuchunks += [self.array.shape[dim]]
                cumuchunks = np.array(cumuchunks)

            # First value greater or equal to
            min_where = np.where(cumuchunks >= self._min_coord[dim])
            if min_where[0].size == 0:
                import pdb

                pdb.set_trace()
            greaterthan_min_idx = (
                min_where[0][0] if min_where[0] is not None else 0
            )
            self._min_coord[dim] = (
                cumuchunks[greaterthan_min_idx - 1]
                if greaterthan_min_idx > 0
                else 0
            )

            max_where = np.where(cumuchunks >= self._max_coord[dim])
            if max_where[0].size == 0:
                import pdb

                pdb.set_trace()
            lessthan_max_idx = (
                max_where[0][0] if max_where[0] is not None else 0
            )
            self._max_coord[dim] = (
                cumuchunks[lessthan_max_idx]
                if lessthan_max_idx < cumuchunks[-1]
                else cumuchunks[-1] - 1
            )

        # Update translate
        self.translate = self._min_coord

        # interval size may be one or more chunks
        interval_size = [
            mx - mn for mx, mn in zip(self._max_coord, self._min_coord)
        ]

        LOGGER.debug(
            f"VirtualData: update_with_minmax: {self.translate} max \
            {self._max_coord} interval size {interval_size}"
        )

        # Update hyperslice
        new_shape = [
            int(mx - mn) for (mx, mn) in zip(self._max_coord, self._min_coord)
        ]

        # Try to reuse the previous hyperslice if possible (otherwise we get
        # flashing) shape of the chunks
        next_hyperslice = np.zeros(new_shape, dtype=self.dtype)

        if prev_max_coord:
            # Get the matching slice from both data planes
            next_slices = []
            prev_slices = []
            for dim in range(len(self._max_coord)):
                # to ensure that start is non-negative
                # prev_start is the start of the overlapping region in the
                # previous one
                if self._min_coord[dim] < prev_min_coord[dim]:
                    prev_start = 0
                    next_start = prev_min_coord[dim] - self._min_coord[dim]
                else:
                    prev_start = self._min_coord[dim] - prev_min_coord[dim]
                    next_start = 0

                width = min(
                    self.hyperslice.shape[dim], next_hyperslice.shape[dim]
                )
                # to make sure its not overflowing the shape
                width = min(
                    width,
                    width
                    - ((next_start + width) - next_hyperslice.shape[dim]),
                    width
                    - ((prev_start + width) - self.hyperslice.shape[dim]),
                )

                prev_stop = prev_start + width
                next_stop = next_start + width

                prev_slices += [slice(int(prev_start), int(prev_stop))]
                next_slices += [slice(int(next_start), int(next_stop))]

            if (
                next_hyperslice[tuple(next_slices)].size > 0
                and self.hyperslice[tuple(prev_slices)].size > 0
            ):
                LOGGER.info(
                    f"reusing data plane: prev {prev_slices} next \
                    {next_slices}"
                )
                if (
                    next_hyperslice[tuple(next_slices)].size
                    != self.hyperslice[tuple(prev_slices)].size
                ):
                    import pdb

                    pdb.set_trace()
                next_hyperslice[tuple(next_slices)] = self.hyperslice[
                    tuple(prev_slices)
                ]
            else:
                LOGGER.info(
                    f"could not reuse data plane: prev {prev_slices} next \
                    {next_slices}"
                )

        self.hyperslice = next_hyperslice

    def _hyperslice_key(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ):
        """Convert a key into a key for the hyperslice.

        The _hyperslice_key is the corresponding value of key
        within plane. The difference between key and hyperslice_key is
        key - translate / scale
        """
        # TODO maybe fine, but a little lazy
        if type(key) is not tuple:
            key = (key,)

        indices = [None] * len(key)
        for i, index in enumerate(indices):
            if type(index) == slice:
                indices[i] = slice(
                    index.start - self.translate[i],
                    index.stop - self.translate[i],
                )
            elif np.issubdtype(int, type(index)):
                indices[i] = index - self.translate[i]
            # else:
            #     LOGGER.info(f"_hyperslice_key: unexpected type {type(index)}\
            #     with value {index}")

        if type(key) is tuple and type(key[0]) == slice:
            if key[0].start is None:
                return key
            hyperslice_key = tuple(
                [
                    slice(
                        int(
                            max(
                                0,
                                sl.start - self.translate[idx],
                            )
                        ),
                        int(
                            max(
                                0,
                                sl.stop - self.translate[idx],
                            )
                        ),
                        sl.step,
                    )
                    for (idx, sl) in enumerate(key[(-self.ndisplay) :])
                ]
            )
        elif type(key) is tuple and type(key[0]) is int:
            hyperslice_key = tuple(
                [
                    int(v - self.translate[idx])
                    for idx, v in enumerate(key[(-self.ndisplay) :])
                ]
            )
        else:
            LOGGER.info(f"_hyperslice_key: funky key {key}")
            hyperslice_key = key

        return hyperslice_key

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Return item from array.

        key is in data coordinates
        """
        return self.get_offset(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Set an item in the array.

        key is in data coordinates
        """
        self.hyperslice[key] = value
        return self.hyperslice[key]

    def get_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Return item from array.

        key is in data coordinates.
        """
        hyperslice_key = self._hyperslice_key(key)
        try:
            return self.hyperslice.__getitem__(hyperslice_key)
        except Exception:
            # if it isnt a slice it is an int so width=1
            shape = tuple(
                [
                    (1 if sl.start is None else sl.stop - sl.start)
                    if type(sl) is slice
                    else 1
                    for sl in key
                ]
            )
            LOGGER.info(f"get_offset failed {key}")
            return np.zeros(shape)

    def set_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Return self[key]."""
        hyperslice_key = self._hyperslice_key(key)
        LOGGER.info(
            f"set_offset: {hyperslice_key} shape in plane: \
            {self.hyperslice[hyperslice_key].shape} value shape: {value.shape}"
        )

        if self.hyperslice[hyperslice_key].size > 0:
            try:
                self.hyperslice[hyperslice_key] = value
            except Exception:
                import pdb

                pdb.set_trace()
        return self.hyperslice[hyperslice_key]

    @property
    def chunksize(self):
        """Return the size of a chunk."""
        if isinstance(self.array, da.Array):
            return self.chunksize
        else:
            return self.array.info

    @property
    def chunks(self):
        """Return the chunks of the array."""
        # TODO this isn't safe because array can be dask or zarr
        return self.array.chunks


class MultiScaleVirtualData:
    """MultiScaleVirtualData encapsulates multiscale arrays.

    The added value is that MultiScaleVirtualData has a small memory
    footprint.

    MultiScaleVirtualData is a parent that tracks the transformation
    between each VirtualData child relative to the highest resolution
    VirtualData (which is just a re-scaling transform). It accepts inputs in
    the coordinate system of the highest resolution VirtualData.

    VirtualData is used to use a 2D array to represent
    a larger shape. The purpose of this function is to provide
    a 2D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.

    NEW: use a translate to define subregion of image

    Attributes
    ----------
    arrays: dask.array
        List of dask arrays of image data, one for each scale
    dtype: dtype
        dtype of the arrays
    shape: tuple
        shape of the true data, the highest resolution (x, y)
    ndim: int
        Number of dimensions, aka scales
    _data: list[VirtualData]
        List of VirtualData objects for each scale
    _translate: list[tuple(int)]
        List of tuples, e.g. [(0, 0), (0, 0)] of translations for each scale
        array
    _scale_factors: list[list[float]]
        List of lists, e.g. [[1.0, 1.0], [2.0, 2.0]], of scale factors between
        the highest resolution scale and each subsequent scale, [x, y]
    ndisplay: int
        number of dimensions to display (equivalent to ndisplay in
        napari.viewer.Viewer)
    _chunk_slices: list of list of chunk slices
        List of list of chunk slices. A slice object for each scale, for each
        dimension,
        e.g. [list of scales[list of slice objects[(x, y, z)]]]
    """

    def __init__(self, arrays, ndisplay=2):
        # TODO arrays should be typed as MultiScaleData
        # each array represents a scale
        self.arrays = arrays
        # first array is considered the highest resolution
        # TODO [kcp] - is that always true?
        highest_res = arrays[0]

        self.dtype = highest_res.dtype
        # This shape is the shape of the true data, but not our hyperslice
        self.shape = highest_res.shape

        self.ndim = len(arrays)

        self.ndisplay = ndisplay

        # Keep a VirtualData for each array
        self._data = []
        self._translate = []
        self._scale_factors = []
        for scale in range(len(self.arrays)):
            virtual_data = VirtualData(
                self.arrays[scale], scale=scale, ndisplay=self.ndisplay
            )
            self._translate += [
                tuple([0] * len(self.shape))
            ]  # Factors to shift the layer by in units of world coordinates.
            # TODO [kcp] there are assumptions here, expect rounding errors
            #      here, or should we force ints?
            self._scale_factors += [
                [
                    hr_val / this_val
                    for hr_val, this_val in zip(
                        highest_res.shape, self.arrays[scale].shape
                    )
                ]
            ]
            self._data += [virtual_data]

    def set_interval(self, min_coord, max_coord, visible_scales=None):
        """Set the extents for each of the scales.

        Extents are set in the coordinates of each individual scale array

        visible_scales must be an empty list of a list equal to the number of
        scales

        Sets the _min_coord and _max_coord on each individual VirtualData

        min_coord and max_coord are in the same units as the highest resolution
        scale data.

        Parameters
        ----------
        min_coord: np.array
            min coordinate in data space, should correspond to top left corner
            of the visible canvas
        max_coord: np.array
            max coordinate in data space, should correspond to bottom right
            corner of the visible canvas
        visible_scales: list
            Optional. ???
        """
        # Bound min_coord and max_coord
        if visible_scales is None:
            visible_scales = []
        max_coord = np.min((max_coord, self._data[0].shape), axis=0)
        min_coord = np.max((min_coord, np.zeros_like(min_coord)), axis=0)

        # for each scale, set the interval for the VirtualData
        # e.g. a high resolution scale may cover [0,1,2,3,4] but a scale
        # with half of that resolution will cover the same region with
        # coords/indices of [0,1,2]
        for scale in range(len(self.arrays)):
            if not visible_scales or visible_scales[scale]:
                # Update translate
                # TODO expect rounding errors here
                scaled_min = [
                    int(min_coord[idx] / self._scale_factors[scale][idx])
                    for idx in range(len(min_coord))
                ]
                scaled_max = [
                    int(max_coord[idx] / self._scale_factors[scale][idx])
                    for idx in range(len(max_coord))
                ]

                self._translate[scale] = scaled_min
                LOGGER.info(
                    f"MultiscaleVirtualData: update_with_minmax: scale {scale}\
                    min {min_coord} : {scaled_min} max {max_coord} : \
                    scaled max {scaled_max}"
                )

                # Ask VirtualData to update its interval
                coords = tuple(
                    [slice(mn, mx) for mn, mx in zip(scaled_min, scaled_max)]
                )
                self._data[scale].set_interval(coords)
            else:
                LOGGER.debug('visible scales are provided, do nothing')
