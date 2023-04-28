import itertools

from typing import Tuple, Union
import dask.array as da
import numpy as np
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.qt.threading import thread_worker


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
    while real_array is None and retry < num_retry:
        try:
            real_array = np.asarray(array[chunk_slice].compute(), dtype=dtype)
            # TODO check for a race condition that is causing this exception
            #      some dask backends are not thread-safe
        except ValueError(
            f"get_chunk failed to fetch data: retry {retry} of {num_retry}"
        ):
            pass
        retry += 1
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


def prioritised_chunk_loading(depth, distance, zoom, alpha=1.0, visible=None):
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
    """
    chunk_load_priority = depth + alpha * zoom * distance
    if visible is not None:
        chunk_load_priority[np.logical_not(visible)] = np.inf
    return chunk_load_priority


# ---------- 3D specific ----------


def chunk_centers_3D(array: da.Array):
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
    # We impose 3D here
    all_middle_pos = [el[-3:] for el in list(itertools.product(*middle_pos))]
    all_end_pos = list(itertools.product(*end_pos))
    chunk_slices = []
    for start, end in zip(all_start_pos, all_end_pos):
        chunk_slice = [
            slice(start_i, end_i) for start_i, end_i in zip(start, end)
        ]
        # We impose 3D here
        chunk_slices.append(tuple(chunk_slice[-3:]))

    mapping = dict(zip(all_middle_pos, chunk_slices))
    return mapping


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
    priorities = prioritised_chunk_loading(
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

def interpolated_get_chunk_2D(
    chunk_slice, array=None
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
    real_array = None
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


class VirtualData:
    """VirtualData is used to use a 2D array to represent
    a larger shape. The purpose of this function is to provide
    a 2D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.

    NEW: use a translate to define subregion of image
    """

    def __init__(self, dtype, shape, scale_factor=(1, 1)):
        self.dtype = dtype
        # This shape is the shape of the true data, but not our data_plane
        self.shape = shape
        self.ndim = len(shape)
        self.translate = tuple([0] * len(shape))
        self.scale_factor = scale_factor

        self.d = 2

        # TODO: I don't like that this is making a choice of slicing axis
        # self.data_plane = np.zeros(self.shape[-1 * self.d :], dtype=self.dtype)
        self.data_plane = np.zeros(1)

    def update_with_minmax(self, min_coord, max_coord):
        # Update translate
        self.translate = min_coord

        print(f"update_with_minmax: min {min_coord} max {max_coord}")

        # Update data_plane
        max_coord = [mx / sf for (mx, sf) in zip(max_coord, self.scale_factor)]
        min_coord = [mn / sf for (mn, sf) in zip(min_coord, self.scale_factor)]

        new_shape = [int(mx - mn) for (mx, mn) in zip(max_coord, min_coord)]
        self.data_plane = np.zeros(new_shape, dtype=self.dtype)

    def _fix_key(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ):
        if type(key) is tuple:
            if key[0].start is None:
                return key
            fixed_key = tuple(
                [
                    slice(
                        max(
                            0,
                            sl.start,
                        ),
                        max(
                            0,
                            sl.stop,
                        ),
                        sl.step,
                    )
                    for (idx, sl) in enumerate(key[-1 * self.d :])
                ]
            )
            val_shape = self.data_plane.__getitem__(fixed_key).shape
            key_size = tuple(
                [
                    slice(0, min((sl.stop - sl.start), fk_val))
                    for sl, fk_val in zip(fixed_key, val_shape)
                ]
            )
            if fixed_key[0].stop == 0:
                import pdb

                pdb.set_trace()
        return fixed_key, key_size

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        fixed_key, _ = self._fix_key(key)
        return self.data_plane.__getitem__(fixed_key)
        # if type(key) is tuple:
        #     return self.data_plane.__getitem__(tuple(key[-1 * self.d :]))
        # else:
        #     return self.data_plane.__getitem__(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        fixed_key, key_size = self._fix_key(key)
        # print(f"virtualdata setitem {key} fixed to {fixed_key}")
        if (
            self.data_plane.__getitem__(fixed_key).shape[0]
            != value[key_size].shape[0]
        ):
            import pdb

            pdb.set_trace()

            # TODO resume here to find out why there are mismatched shapes after update)with_min_max

        # TODO trim key_size because min_max size is based on screen and is ragged

        return self.data_plane.__setitem__(fixed_key, value[key_size])
        # if type(key) is tuple:
        #     return self.data_plane.__setitem__(
        #         tuple(key[-1 * self.d :]), value
        #     )
        # else:
        #     return self.data_plane.__setitem__(key, value)

