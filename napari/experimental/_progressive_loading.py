import heapq
import itertools
import logging
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Union

import dask.array as da
import numpy as np
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread

from napari._vispy.utils.gl import get_max_texture_sizes
from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
)
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


def chunk_slices(
    array: Union[da.Array, np.ndarray], interval: Optional[Iterable] = None
) -> List[List[slice]]:
    array = array.array
    """Create a list of slice objects for each chunk for each dimension."""
    if isinstance(array, da.Array):
        # For Dask Arrays
        start_pos = [np.cumsum(sizes) - sizes for sizes in array.chunks]
        end_pos = [np.cumsum(sizes) for sizes in array.chunks]

    else:
        # For Zarr Arrays
        start_pos = []
        end_pos = []
        for dim in range(len(array.chunks)):
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
            cumuchunks = list(
                range(int(start_idx), int(stop_idx), array.chunks[dim])
            )
            cumuchunks = np.array(cumuchunks)
            start_pos += [cumuchunks[:-1]]
            end_pos += [cumuchunks[1:]]

    if interval is not None:
        for dim in range(len(start_pos)):
            first_idx = np.searchsorted(end_pos[dim], interval[0, dim])
            last_idx = np.searchsorted(
                start_pos[dim], interval[1, dim], side='right'
            )
            start_pos[dim] = start_pos[dim][first_idx:last_idx]
            end_pos[dim] = end_pos[dim][first_idx:last_idx]

    chunk_slices = [[] for _ in range(len(array.chunks))]
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


def draw_chunk_boundaries(
    corner_pixels, data, layer=None, viewer=None, scale=None
):
    """
    This draws cubes around all chunks within the corner pixels
    """
    chunk_keys = chunk_slices(data, interval=corner_pixels)

    edge_lookup = np.array(
        [
            [0, 1],
            [1, 3],
            [3, 2],
            [2, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [5, 7],
            [7, 6],
            [6, 4],
        ]
    )

    all_edges = []

    chunk_coords = itertools.product(*chunk_keys)
    for chunk_coord in chunk_coords:
        top_left = [sl.start for sl in chunk_coord]
        bottom_right = [sl.stop for sl in chunk_coord]

        bounds = list(zip(top_left, bottom_right))

        vertices = np.array(list(itertools.product(*bounds))) * np.array(scale)

        edges = [
            [list(vertices[index[0]]), list(vertices[index[1]])]
            for index in edge_lookup
        ]

        all_edges += edges

    if layer is None:
        layer = viewer.add_shapes(
            all_edges, shape_type="line", edge_width=scale[0] / 10
        )
    else:
        layer.data = all_edges

    return layer


@tz.curry
def progressively_update_layer(invar, viewer, data=None, ndisplay=None):
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

    if (
        "should_render" in root_layer.metadata
        and not root_layer.metadata["should_render"]
    ):
        LOGGER.info(f"Not rendering because {root_layer.metadata}")
        return

    worker = None
    if "worker" in root_layer.metadata:
        worker = root_layer.metadata["worker"]

    if "MultiScaleVirtualData" in root_layer.metadata:
        data = root_layer.metadata["MultiScaleVirtualData"]

    # TODO global worker usage is not viable for real implementation
    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.await_workers()
        # worker.await_workers(msecs=30000)

    # Find the corners of visible data in the highest resolution
    corner_pixels = root_layer.corner_pixels

    LOGGER.info(f"corner pixels: {corner_pixels}")

    top_left = corner_pixels[0, :]
    bottom_right = corner_pixels[1, :]

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
        f"progressively_update_layer: start render_sequence {corner_pixels} on {root_layer}"
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

        # TODO make this optional to support debugging
        # if visible_scales[scale]:
        #     debug_layer = None
        #     if "debug_layer" in layer.metadata:
        #         debug_layer = layer.metadata["debug_layer"]
        #         debug_layer.visible = True
        #     debug_layer = draw_chunk_boundaries(
        #         corner_pixels / data._scale_factors[scale],
        #         layer.data,
        #         layer=debug_layer,
        #         viewer=viewer,
        #         scale=data._scale_factors[scale],
        #     )
        #     layer.metadata["debug_layer"] = debug_layer
        # else:
        #     if "debug_layer" in layer.metadata:
        #         layer.metadata["debug_layer"].visible = False

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
        f"progressively_update_layer: started render_sequence with corners {corner_pixels}"
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

        # Log the shape of the array being set as texture data
        if layer.data.hyperslice.size == 0 or any(
            dim == 0 for dim in layer.data.hyperslice.shape
        ):
            LOGGER.warning(
                f"Trying to set empty array as texture data: Shape {layer.data.hyperslice.shape}"
            )

        layer.data.set_offset(chunk_slice, chunk)
        texture.set_data(layer.data.hyperslice)
        node.update()

    worker.yielded.connect(on_yield)
    root_layer.metadata["worker"] = worker
    worker.start()


def initialize_multiscale_virtual_data(img, viewer, ndisplay):
    """Initialize MultiScaleVirtualData and set interval.

    This function also enforces GL memory constraints.
    """

    #

    multiscale_data = MultiScaleVirtualData(img, ndisplay=ndisplay)

    max_size = get_max_texture_sizes()[ndisplay - 2]

    # Get initial extent for rendering
    canvas_corners = (
        viewer.window._qt_viewer.canvas._canvas_corners_in_world.copy()
    )
    canvas_corners[canvas_corners < 0] = 0
    canvas_corners = canvas_corners.astype(np.int64)

    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]

    if np.any(bottom_right < top_left):
        LOGGER.warning(
            f"Issue with bottom_right values detected, returning early. top_left {top_left} and bottom_right {bottom_right}"
        )
        return None

    if max_size is not None:
        # Bound the interval with the maximum texture size
        for i in range(len(top_left)):
            bottom_right[i] = min(bottom_right[i], top_left[i] + max_size)

    if ndisplay != len(img[0].shape):
        top_left = [viewer.dims.point[-ndisplay]] + top_left.tolist()
        bottom_right = [viewer.dims.point[-ndisplay]] + bottom_right.tolist()

    multiscale_data.set_interval(top_left, bottom_right)

    # TODO this goes away when we stop using multiple layers
    if get_layer_name_for_scale(0) in viewer.layers:
        root_layer = viewer.layers[get_layer_name_for_scale(0)]
        root_layer.metadata["MultiScaleVirtualData"] = multiscale_data

    return multiscale_data


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
    if contrast_limits is None:
        contrast_limits = [0, 255]

    if not viewer:
        from napari import Viewer

        viewer = Viewer()

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"
    viewer.dims.ndim = ndisplay
    viewer._layer_slicer._force_sync = False

    # Call the helper function to initialize MultiScaleVirtualData
    multiscale_data = initialize_multiscale_virtual_data(img, viewer, ndisplay)
    LOGGER.info(f"Adding MultiscaleData with shape: {multiscale_data.shape}")

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
        scale = np.ones(ndisplay)
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

    # Connect to camera and dims
    for listener in [viewer.camera.events, viewer.dims.events.current_step]:
        listener.connect(
            debounced(
                ensure_main_thread(
                    progressively_update_layer(
                        data=multiscale_data,
                        viewer=viewer,
                        ndisplay=ndisplay,
                    )
                ),
                timeout=2000,
            )
        )

    # Trigger first render
    progressively_update_layer(
        None, data=multiscale_data, viewer=viewer, ndisplay=ndisplay
    )

    layers[0].metadata["should_render"] = True

    return viewer


# ---------- 2D specific ----------


def chunk_priority_2D(
    chunk_keys, corner_pixels, scale_factor, center_weight=5
):
    """
    Return a priority map of chunk keys within the corner_pixels at a specific
    scale level.

    The function calculates the priority of each chunk based on its distance
    from the center of the view. The priority is influenced by the provided
    `center_weight` parameter to emphasize the loading of chunks near the
    center of the view.

    Parameters
    ----------
    chunk_keys : list of list of slices for each dimension
        A list of list of slices representing the chunk keys for each dimension
    corner_pixels : tuple of np.ndarray
        2D top left and bottom right coordinates for the current view in the
        format (top_left_coords, bottom_right_coords), where each is a 1-D
        array of length 2.
    scale_factor : float
        The scale factor for this scale level.
    center_weight : float, optional
        A weight factor to control the influence of the distance from the
        center of the  view in the priority calculation. Higher values give
        more weight to the centrality of the chunk. Default is 5.

    Returns
    -------
    priority_map : list of tuple
        A priority map represented as a list of tuples, where each tuple
        contains the priority value and the corresponding chunk key. The list
        is sorted by priority.

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
        # Calculate the center of each chunk
        chunk_center = get_chunk_center(chunk_key)

        # Calculate the distance from chunk center to the center of the view
        view_center = np.mean(corner_pixels, axis=0) / scale_factor
        center_view_dist = np.linalg.norm(chunk_center - view_center)

        # Calculate priority based on center_view_dist and center_weight
        priority = (
            center_weight * center_view_dist * (1 + 0.0001 * np.random.rand())
        )

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


def chunk_priority_3D(
    chunk_keys, corner_pixels, scale_factor, camera, center_weight=5
):
    """Return a priority map of chunk keys within the corner_pixels at a
    specific scale level.

    The function calculates the priority of each chunk based on its visual
    depth, distance from the camera's center line, and distance from the center
    of the camera's view. The priority is influenced by the provided
    `center_weight` parameter to emphasize the loading of chunks near the
    center of the camera's view.

    Parameters
    ----------
    chunk_keys : list of list of slices for each dimension
        A list of list of slices representing the chunk keys for each dimension
    corner_pixels : tuple of np.ndarray
        ND top left and bottom right coordinates for the current view in the
        format (top_left_coords, bottom_right_coords), where each is a 1-D
        array of length N.
    scale_factor : float
        The scale factor for this scale level.
    camera : object
        A camera object containing necessary camera parameters such as zoom
        level.
    center_weight : float, optional
        A weight factor to control the influence of the distance from the
        center of the camera's view in the priority calculation. Higher values
        give more weight to the centrality of the chunk. Default is 5.

    Returns
    -------
    priority_map : list of tuple
        A priority map represented as a list of tuples, where each tuple
        contains the priority value and the corresponding chunk key. The
        list is sorted by priority.

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

        # New measure: distance from chunk center to the center of the camera's view
        camera_center = np.mean(corner_pixels, axis=0) / scale_factor
        center_view_dist = np.linalg.norm(chunk_center - camera_center)

        # Define weights for each factor
        depth_weight = 1
        center_line_weight = camera.zoom

        # Updated priority calculation
        priority = (
            depth_weight * depth
            + center_line_weight * center_line_dist
            + center_weight * center_view_dist
        ) * (1 + 0.0001 * np.random.rand())

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

