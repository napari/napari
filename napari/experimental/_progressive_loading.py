import time
import heapq
import itertools
import logging
import sys
from collections import defaultdict
from typing import Tuple, Union

import zarr
import dask
import dask.array as da
import numpy as np

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


class ChunkFailed(Exception):
    pass


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
        # try:
        if True:
            if isinstance(array, zarr.Array):
                coords = tuple([int(sl.start / array._chunks[dim]) for dim, sl in enumerate(chunk_slice)])
                real_array = array.get_chunk(coords)
            else:
                # For Dask fetching
                chunk = array[chunk_slice]
                LOGGER.info(f"get_chunk (sliced) : {(time.time() - start_time)}")

                if isinstance(chunk, da.Array) or isinstance(chunk, VirtualData):
                    real_array = chunk.compute()
                    LOGGER.info(f"get_chunk (compute) : {(time.time() - start_time)} type {type(chunk)}")
                else:
                    # real_array = np.asarray(chunk, dtype=dtype)
                    real_array = chunk

            # TODO check for a race condition that is causing this exception
            #      some dask backends are not thread-safe
        # except ChunkFailed(
        #     f"get_chunk failed to fetch data: retry {retry} of {num_retry}"
        # ):
        #     pass
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


def distance_from_camera_centre_line(points, camera):
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


def chunk_slices(array: da.Array, ndim=3, interval=None):
    """Create a list of slice objects for each chunk for each dimension. 

    Make a dictionary mapping chunk centers to chunk slices.
    Note: if array is >3D, then the last 3 dimensions are assumed as ZYX
    and will be used for calculating centers. If array is <3D, the third 
    dimension is assumed to be None.


    Parameters
    ----------
    array: dask or zarr Array
        The input array, a single scale

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
                start_idx = np.floor(interval[0,dim] / array.chunks[dim]) * array.chunks[dim]
                stop_idx = np.ceil(interval[1,dim] / array.chunks[dim]) * array.chunks[dim] + 1
            # Inclusive on the end point
            cumuchunks = [val for val in range(int(start_idx), int(stop_idx), array.chunks[dim])]
            cumuchunks = np.array(cumuchunks)
            start_pos += [cumuchunks[:-1]]
            end_pos += [cumuchunks[1:]]
        
    if interval is not None:
        # array([[7709.88671875, 5007.1953125 ],[9323.7578125 , 6824.38867188]])
        #
        for dim in range(len(start_pos)):
            # Find first index in end_pos that is greater than corner_pixels
            first_idx = np.searchsorted(end_pos[dim], interval[0, dim])
            # Find the last index in start_pos that is less than corner_pixels[1,dim]
            last_idx = np.searchsorted(
                start_pos[dim], interval[1, dim], side='right'
            )

            start_pos[dim] = start_pos[dim][first_idx:last_idx]
            end_pos[dim] = end_pos[dim][first_idx:last_idx]

    all_start_pos = list(itertools.product(*start_pos))
    # TODO We impose dimensional ordering for ND
    all_end_pos = list(itertools.product(*end_pos))
    chunk_slices = []

    chunk_slices = [[]] * len(array.chunks)
    for dim in range(len(array.chunks)):
        chunk_slices[dim] = [
            slice(st, end) for st, end in zip(start_pos[dim], end_pos[dim])
        ]

    return chunk_slices


# ---------- 2D specific ----------


def chunk_priority_2D(chunk_keys, corner_pixels, scale):
    """Return the keys for all chunks at this scale within the corner_pixels

    Parameters
    ----------
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    scale : int
        the scale level, assuming powers of 2

    """

    # NOTE chunk_keys is a list of lists of slices for each dimension

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code

    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    # contained_keys is an array with list of slices contained along each dimension
    contained_keys = defaultdict(list)
    for dim, chunk_slices in enumerate(chunk_keys):
        for sl in chunk_slices:
            below_min = sl.start < mins[dim]
            above_max = sl.stop > maxs[dim]
            # If start and stop are below interval, or
            #    start and stop are above interval: return False
            if (below_min and sl.stop < mins[dim]) or (
                above_max and sl.start > maxs[dim]
            ):
                pass
            else:
                contained_keys[dim] += [sl]

    priority_map = []

    for idx, chunk_key in enumerate(
        list(
            itertools.product(
                *[contained_keys[k] for k in sorted(contained_keys.keys())]
            )
        )
    ):
        priority = 0
        # TODO filter priority here
        if True:
            priority = 0
        else:
            priority = np.inf
        if priority < np.inf:
            heapq.heappush(priority_map, (priority, chunk_key))

    return priority_map


# ---------- 3D specific ----------


def prioritised_chunk_loading_3D(
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
    scale_factors=[],
    dtype=np.uint16,
    dims=None,
):
    """
    Entry point for recursive function render_sequence.

    See render_sequence for docs.
    """
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
    scale_factors=[],
    dtype=np.uint16,
    dims=None,
):
    """Recursively add multiscale chunks to a napari viewer for some multiscale arrays

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
        see prioritised_chunk_loading for more info
    scale_factors : list of tuples
        a list of tuples of scale factors for each array
    dtype : dtype
        dtype of data
    """

    layer_name = f"{container}/{dataset}/s{scale}"

    # Get some variables specific to this scale level
    min_coord = [st.start for st in view_slice]
    max_coord = [st.stop for st in view_slice]
    array = arrays[scale]
    chunk_map = chunk_maps[scale]
    scale_factor = scale_factors[scale]

    highres_min = str([el.start * 2**scale for el in view_slice])
    highres_max = str([el.stop * 2**scale for el in view_slice])

    # LOGGER.info(
    #     f"add_subnodes {scale} {str(view_slice)}",
    #     f"highres interval: {highres_min},  {highres_max}",
    #     f"chunksize: {array.chunksize} arraysize: {array.shape}",
    # )

    # Points for each chunk, for example, centers
    points = np.array(list(chunk_map.keys()))

    # Mask of whether points are within our interval, this is in array coordinates
    point_mask = np.array(
        [
            np.all(point >= min_coord) and np.all(point <= max_coord)
            for point in points
        ]
    )

    # Rescale points to world for priority calculations
    points_world = points * np.array(scale_factor)

    # Prioritize chunks using world coordinates
    distances = distance_from_camera_centre_line(points_world, camera)
    depth = visual_depth(points_world, camera)
    priorities = prioritised_chunk_loading_3D(
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

    # This node's offset in world space
    world_offset = np.array(min_coord) * np.array(scale_factor)

    # Iterate over points/chunks and add corresponding nodes when appropriate
    for idx, point in enumerate(points):
        # TODO: There are 2 strategies here:
        # 1. Render *visible* chunks, or all if we're on the last scale level
        #    Skip the chunk at this resolution because it will be shown in higher res
        #    This fetches less data.
        # if point_mask[idx] and (idx not in best_priorities or scale == 0):
        # 2. Render all chunks because we know we will zero out this data when
        #    it is loaded at the next resolution level.
        if point_mask[idx]:
            coord = tuple(point)
            chunk_slice = chunk_map[coord]
            offset = [sl.start for sl in chunk_slice]
            min_interval = offset

            # find position and scale
            node_offset = (
                min_interval[0] * scale_factor[0],
                min_interval[1] * scale_factor[1],
                min_interval[2] * scale_factor[2],
            )

            # LOGGER.debug(
            #     f"Fetching: {(scale, chunk_slice)} World offset: {node_offset}"
            # )

            # When we get_chunk chunk_slice needs to be in data space, but chunk slices are 3D
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
                # world_offset,
                None,
                chunk_slice,
                texture_slice,
            )

    # TODO make sure that all of low res loads first
    # TODO take this 1 step further and fill all high resolutions with low res

    # recurse on best priorities
    if scale > 0:
        # The next priorities for loading in higher resolution are the best ones
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

            # TODO Note that we need to be blanking out lower res data at the same time
            # TODO this is when we should move the node from the next resolution.
            yield (
                np.asarray(zdata),
                scale,
                tuple([sl.start for sl in chunk_slice]),
                next_world_offset,
                chunk_slice,
                texture_slice,
            )

            # LOGGER.info(
            #     f"Recursive add on\t{str(next_chunk_slice)} idx {priority_idx}",
            #     f"visible {point_mask[priority_idx]} for scale {scale} to {scale-1}\n",
            #     f"Relative scale factor {relative_scale_factor}",
            # )
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


# Code from an earlier stage to support visual debugging
# @tz.curry
# def update_point_colors(event, viewer, alpha=1.0):
#     """Update the points based on their distance to current camera.

#     Parameters:
#     -----------
#     viewer : napari.Viewer
#         Current viewer
#     event : camera.events.angles event
#         The event triggered by changing the camera angles
#     """
#     # TODO we need a grid for each scale, or the grid needs to include all scales
#     points_layer = viewer.layers['grid']
#     points = points_layer.data
#     distances = distance_from_camera_centre_line(points, viewer.camera)
#     depth = visual_depth(points, viewer.camera)
#     priorities = prioritised_chunk_loading(
#         depth, distances, viewer.camera.zoom, alpha=alpha
#     )
#     points_layer.features = pd.DataFrame(
#         {'distance': distances, 'depth': depth, 'priority': priorities}
#     )
#     # TODO want widget to change color
#     points_layer.face_color = 'priority'
#     points_layer.refresh()


# @tz.curry
# def update_shown_chunk(event, viewer, chunk_map, array, alpha=1.0):
#     """
#     chunk map is a dictionary mapping chunk centers to chunk slices
#     array is the array containing the chunks
#     """
#     # TODO hack here to insert the recursive drawing
#     points = np.array(list(chunk_map.keys()))
#     distances = distance_from_camera_centre_line(points, viewer.camera)
#     depth = visual_depth(points, viewer.camera)
#     priorities = prioritised_chunk_loading(
#         depth, distances, viewer.camera.zoom, alpha=alpha
#     )
#     first_priority_idx = np.argmin(priorities)
#     first_priority_coord = tuple(points[first_priority_idx])
#     chunk_slice = chunk_map[first_priority_coord]
#     offset = [sl.start for sl in chunk_slice]
#     # TODO note that this only updates the highest resolution
#     hi_res_layer = viewer.layers['high-res']
#     hi_res_layer.data = array[chunk_slice]
#     hi_res_layer.translate = offset
#     hi_res_layer.refresh()


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
                ((1 - w) * lvalue + w * rvalue).squeeze()
#                ((1 - w) * lvalue + w * rvalue).astype(np.uint16).squeeze()
            )

    LOGGER.info(f"interpolated_get_chunk_2D : {(time.time() - start_time)}")

    return real_array


# TODO a VirtualData should only address 1 coordinate system
class VirtualData:
    """`VirtualData` wraps a particular scale level's array. It acts like an 
    array of that size, but only works within the interval setup by 
    `set_interval`. Each `VirtualData` uses the scale level's coordinates.

    -- `VirtualData` uses a (poorly named) "data_plane" to store the currently 
    active interval.
    -- `VirtualData.translate` specifies the offset of the 
    `VirtualData.data_plane`'s origin from `VirtualData.array`'s origin, in 
    `VirtualData`'s coordinate system
    
    VirtualData is used to use a 2D array to represent
    a larger shape. The purpose of this function is to provide
    a 2D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.

    NEW: use a translate to define subregion of image

    Attributes
    ----------
    array: dask.array
        Dask array of image data for this scale
    dtype: dtype
        dtype of the array
    shape: tuple
        shape of the true data (not the data_plane)
    ndim: int
        Number of dimensions for this scale
    translate: list[tuple(int)]
        tuple for the translation
    d: int
        Dimension of the chunked slices ??? Hard coded to 2.
    data_plane: dask.array
        Array of currently visible data for this layer
    _min_coord: list 
        List of the minimum coordinates in each dimension
    _max_coord: list
        List of the maximum coordinates in each dimension

    """

    def __init__(self, array, scale):
        self.array = array
        self.dtype = array.dtype
        # This shape is the shape of the true data, but not our data_plane
        self.shape = array.shape
        self.ndim = len(self.shape)

        # translate is in the same units as the highest resolution scale
        self.translate = tuple([0] * len(self.shape))

        self.data_plane = da.zeros(1)

        self.d = 2
        self._max_coord = None
        self._min_coord = None
        self.scale = scale  # for debugging

    def set_interval(self, coords):
        """The interval is the range of the data for this scale that is 
        currently visible. It is stored in `_min_coord` and `_max_coord` in
        the coordinates of the original array. 

        This function takes a slice, converts it to a range (stored as 
        `_min_coord` and `_max_coord`), and extracts a subset of the orginal 
        data array (stored as `data_plane`)

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
                cumuchunks = [val for val in range(self.chunks[dim], self.array.shape[dim], self.chunks[dim])]
                # Add last element
                cumuchunks += [self.array.shape[dim]]
                cumuchunks = np.array(cumuchunks)
                

            # First value greater or equal to
            min_where = np.where(cumuchunks > self._min_coord[dim])
            greaterthan_min_idx = min_where[0][0] if min_where[0] is not None else 0
            self._min_coord[dim] = (
                cumuchunks[greaterthan_min_idx - 1]
                if greaterthan_min_idx > 0
                else 0
            )

            
            max_where = np.where(cumuchunks >= self._max_coord[dim])
            lessthan_max_idx = max_where[0][0] if max_where[0] is not None else 0
            self._max_coord[dim] = (
                cumuchunks[lessthan_max_idx]
                if lessthan_max_idx < cumuchunks[-1]
                else cumuchunks[-1] - 1
            )

        # Update translate
        # TODO there is a bug here, _min_coord goes to 0 when it shouldnt as the user zooms into the highest resolution
        # if self.array.shape[0] > 100000 and self._min_coord[0] == 0:
        #     import pdb; pdb.set_trace()
        self.translate = self._min_coord

        # interval size may be one or more chunks
        interval_size = [mx - mn for mx, mn in zip(self._max_coord, self._min_coord)]
        
        LOGGER.debug(f"VirtualData: update_with_minmax: {self.translate} max {self._max_coord} interval size {interval_size}")

        # Update data_plane

        new_shape = [
            int(mx - mn) for (mx, mn) in zip(self._max_coord, self._min_coord)
        ]

        # Try to reuse the previous data_plane if possible (otherwise we get flashing)
        # shape of the chunks
        next_data_plane = np.zeros(new_shape, dtype=self.dtype)

        if prev_max_coord:
            # Get the matching slice from both data planes
            next_slices = []
            prev_slices = []
            for dim in range(len(self._max_coord)):
                # to ensure that start is non-negative
                # prev_start is the start of the overlapping region in the previous one
                if self._min_coord[dim] < prev_min_coord[dim]:
                    prev_start = 0
                    next_start = prev_min_coord[dim] - self._min_coord[dim]
                else:
                    prev_start = self._min_coord[dim] - prev_min_coord[dim]
                    next_start = 0

                # Get smallest width, this is overlap
                # width = min(self._max_coord[dim] - next_start, prev_max_coord[dim] - prev_start)
                # width = min(self._max_coord[dim], prev_max_coord[dim]) - max(next_start, prev_start)
                width = min(
                    self.data_plane.shape[dim], next_data_plane.shape[dim]
                )
                # to make sure its not overflowing the shape
                width = min(
                    width,
                    width
                    - ((next_start + width) - next_data_plane.shape[dim]),
                    width
                    - ((prev_start + width) - self.data_plane.shape[dim]),
                )

                prev_stop = prev_start + width
                next_stop = next_start + width

                # if self._max_coord[dim] < prev_max_coord[dim]:
                #     next_stop = self._max_coord[dim]
                #     prev_stop = prev_start + (next_stop - next_start)
                # else:
                #     prev_stop = prev_max_coord[dim]
                #     next_stop = next_start + (prev_stop - prev_start)

                # prev_start = max(prev_next_offset, self._min_coord[dim] - prev_min_coord[dim])
                # prev_stop = min(prev_start + new_shape[dim], self.data_plane.shape[dim])

                # next_start = 0
                # next_stop = min(next_start + new_shape[dim], self.data_plane.shape[dim])

                prev_slices += [slice(int(prev_start), int(prev_stop))]
                next_slices += [slice(int(next_start), int(next_stop))]

            if (
                next_data_plane[tuple(next_slices)].size > 0
                and self.data_plane[tuple(prev_slices)].size > 0
            ):
                LOGGER.info(
                    f"reusing data plane: prev {prev_slices} next {next_slices}"
                )
                if (
                    next_data_plane[tuple(next_slices)].size
                    != self.data_plane[tuple(prev_slices)].size
                ):
                    import pdb

                    pdb.set_trace()
                next_data_plane[tuple(next_slices)] = self.data_plane[
                    tuple(prev_slices)
                ]
            else:
                LOGGER.info(
                    f"could not data plane: prev {prev_slices} next {next_slices}"
                )

        self.data_plane = next_data_plane

    def _data_plane_key(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ):
        """The _data_plane_key is the corresponding value of key
        within plane. The difference between key and data_plane_key is
        key - translate / scale"""
        if type(key) is tuple and type(key[0]) is slice:
            if key[0].start is None:
                return key
            data_plane_key = tuple(
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
                    for (idx, sl) in enumerate(key[-1 * self.d :])
                    # TODO check out the self.d stuff here and elsewhere
                ]
            )
            # Removed a breakpoint for catching empty slices
        elif type(key) is tuple and type(key[0]) is int:
            data_plane_key = tuple(
                [
                    int(v - self.translate[idx])
                    for idx, v in enumerate(key[-1 * self.d :])
                ]
            )

        return data_plane_key

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        # return self.data_plane.__getitem__(key)
        return self.get_offset(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        self.data_plane[key] = value
        return self.data_plane[key]

    def get_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        data_plane_key = self._data_plane_key(key)
        try:
            return self.data_plane.__getitem__(data_plane_key)
        except Exception:
            shape = tuple([sl.stop - sl.start for sl in key])
            LOGGER.info(f"get_offset failed {key}")
            import pdb

            pdb.set_trace()
            return np.zeros(shape)
        # if type(key) is tuple:
        #     return self.data_plane.__getitem__(tuple(key[-1 * self.d :]))
        # else:
        #     return self.data_plane.__getitem__(key)

    def set_offset(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        data_plane_key = self._data_plane_key(key)
        LOGGER.info(
            f"set_offset: {data_plane_key} shape in plane: {self.data_plane[data_plane_key].shape} value shape: {value.shape}"
        )
        # if np.any(np.array(self.data_plane[data_plane_key].shape) == 0):
        #     import pdb; pdb.set_trace()
        # TODO this is evil
        if self.data_plane[data_plane_key].size > 0:
            self.data_plane[data_plane_key] = value
        return self.data_plane[data_plane_key]
        # if type(key) is tuple:
        #     return self.data_plane.__setitem__(
        #         tuple(key[-1 * self.d :]), value
        #     )
        # else:
        #     return self.data_plane.__setitem__(key, value)

    @property
    def chunksize(self):
        return self.array.info

    @property
    def chunks(self):
        return self.array.chunks


def test_virtualdata():
    shape = (100, 100)

    data = da.from_delayed(
        dask.delayed(lambda: None), shape=shape, dtype=np.int16
    ).rechunk(chunks=(10, 10))

    virtual_data = VirtualData(data, scale=0)

    interval = (slice(10, 20), slice(10, 20))

    virtual_data.set_interval(interval)
    dim = 1

    # Put column indices along rows
    with np.nditer(
        virtual_data[interval], flags=['multi_index'], op_flags=['writeonly']
    ) as it:
        for x in it:
            x[...] = it.multi_index[dim] + virtual_data.translate[dim]

    # Check if the right value is in the offset region
    assert virtual_data[15, 15] == 15

    # Check that data_plane is aligned to the input array's chunks
    interval = (slice(15, 25), slice(15, 25))
    virtual_data.set_interval(interval)

    assert np.all(np.array(virtual_data.translate) == np.array((10, 10)))
    assert np.all(
        np.array(virtual_data.data_plane.shape) == np.array((20, 20))
    )

    print(virtual_data[interval])

    import pdb

    pdb.set_trace()


# TODO MultiScaleVirtualData describes a multiscale relationship between
# multiple VirtualData, including tracking coordinate systems
class MultiScaleVirtualData:
    """MultiScaleVirtualData is a parent that tracks the transformation 
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
    d: int
        Dimension of the chunked slices ???
    _chunk_slices: list of list of chunk slices
        List of list of chunk slices. A slice object for each scale, for each 
        dimension, 
        e.g. [list of scales[list of slice objects[(x, y, z)]]]
    """

    def __init__(self, arrays):
        # TODO arrays should be typed as MultiScaleData
        # each array represents a scale
        self.arrays = arrays
        # first array is considered the highest resolution # TODO [kcp] - is that always true?
        highest_res = arrays[0]

        self.dtype = highest_res.dtype
        # This shape is the shape of the true data, but not our data_plane
        self.shape = highest_res.shape
        self.ndim = len(arrays)

        # Keep a VirtualData for each array
        self._data = []
        self._translate = []
        self._scale_factors = []
        for scale in range(len(self.arrays)):
            virtual_data = VirtualData(self.arrays[scale], scale=scale)
            self._translate += [tuple([0] * len(self.shape))] # Factors to shift the layer by in units of world coordinates.
            # TODO [kcp] there are assumptions here, expect rounding errors here, or should we force ints?
            self._scale_factors += [
                [
                    hr_val / this_val
                    for hr_val, this_val in zip(
                        highest_res.shape, self.arrays[scale].shape
                    )
                ]
            ]
            self._data += [virtual_data]

        # TODO hard coded 2D for now [kcp] what is this?
        self.d = 2

        # This is expensive to precompute for large arrays
        # self._chunk_slices = []
        # for scale, array in enumerate(self.arrays):
        #     print(f"init of {scale}")
        #     these_slices = chunk_slices(array, ndim=self.d)
        #     self._chunk_slices += [these_slices]

    def set_interval(self, min_coord, max_coord, visible_scales=[]):
        """Set the extents for each of the scales in the coordinates of each
        individual scale array

        visible_scales must be an empty list of a list equal to the number of 
        scales

        Sets the _min_coord and _max_coord on each individual VirtualData

        min_coord and max_coord are in the same units as the highest resolution
        scale data.

        Parameters
        ----------
        min_coord: np.array
            top left corner of the visible canvas
        max_coord: np.array
            bottom right corner of the visible canvas
        visible_scales: list
            Optional. ???
        """

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
                    f"MultiscaleVirtualData: update_with_minmax: scale {scale} min {min_coord} : {scaled_min} max {max_coord} : scaled max {scaled_max}"
                )

                # Ask VirtualData to update its interval
                coords = tuple(
                    [slice(mn, mx) for mn, mx in zip(scaled_min, scaled_max)]
                )
                self._data[scale].set_interval(coords)
            else:
                LOGGER.debug('visible scales are provided, do nothing')


if __name__ == "__main__":
    test_virtualdata()
