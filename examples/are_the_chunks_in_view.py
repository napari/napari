import itertools
import logging
import os
import sys
import threading
import time

import dask.array as da
import numpy as np
import toolz as tz

from magicgui import magic_factory
from psygnal import debounced
from scipy.spatial.transform import Rotation as R
from superqt import ensure_main_thread

import napari
from napari.qt.threading import thread_worker

from napari.experimental._progressive_loading import (
    ChunkCacheManager,
    luethi_zenodo_7144919,
    get_chunk,
)

LOGGER = logging.getLogger("poor-mans-octree")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)


def chunk_centers(array: da.Array):
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


def rotation_matrix_from_camera(
    angles,
) -> np.ndarray:
    return R.from_euler(seq='yzx', angles=angles, degrees=True)


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


@thread_worker
def render_sequence_caller(
    view_slice,
    scale=0,
    camera=None,
    cache_manager=None,
    arrays=None,
    chunk_maps=None,
    container="",
    dataset="",
    alpha=0.8,
    scale_factors=[],
    dtype=np.uint16,
    dims=None,
):
    """
    Entry point for recursive function render_sequence.

    See render_sequence for docs.
    """
    yield from render_sequence(
        view_slice,
        scale=scale,
        camera=camera,
        cache_manager=cache_manager,
        arrays=arrays,
        chunk_maps=chunk_maps,
        container=container,
        dataset=dataset,
        alpha=alpha,
        scale_factors=scale_factors,
        dtype=dtype,
        dims=dims,
    )


def render_sequence(
    view_slice,
    scale=0,
    camera=None,
    cache_manager=None,
    arrays=None,
    chunk_maps=None,
    container="",
    dataset="",
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

            scale_dataset = f"{dataset}/s{scale}"

            # When we get_chunk chunk_slice needs to be in data space, but chunk slices are 3D
            data_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [slice(sl.start, sl.stop) for sl in chunk_slice]
            )

            data = get_chunk(
                data_slice,
                array=array,
                container=container,
                dataset=scale_dataset,
                cache_manager=cache_manager,
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
            yield from render_sequence(
                next_chunk_slice,
                scale=scale - 1,
                camera=camera,
                cache_manager=cache_manager,
                arrays=arrays,
                chunk_maps=chunk_maps,
                container=container,
                dataset=dataset,
                scale_factors=scale_factors,
                dtype=dtype,
                dims=dims,
            )


# TODO consider filling these chunks into a queue and processing them in batches
@tz.curry
def update_chunk(
    chunk_tuple,
    viewer_lock=None,
    viewer=None,
    container="",
    dataset="",
    dtype=np.uint8,
):
    """Update the display with a chunk

    Update a display when a chunk is recieved. This has been developed
    as a function that is associated with on_yielded events from a
    generator that yields chunks of image data (e.g. chunks from a
    zarr image).

    Parameters
    ----------
    chunk_tuple : tuple
        tuple that contains data and metadata necessary to update the
        display
    viewer : napari.viewer.Viewer
        a napari viewer with layers the given container, dataset that
        will be updated with the chunk's data
    container : str
        a zarr container
    dataset : str
        group in container
    dtype : dtype
        a numpy-like dtype

    """
    with viewer_lock:
        tic = time.perf_counter()
        (
            data,
            scale,
            array_offset,
            node_offset,
            chunk_slice,
            texture_slice,
        ) = chunk_tuple

        # TODO 3D assumed here as last 3 dimensions
        texture_offset = tuple([sl.start for sl in texture_slice[-3:]])

        layer_name = f"{container}/{dataset}/s{scale}"
        layer = viewer.layers[layer_name]
        volume = viewer.window.qt_viewer.layer_to_visual[
            layer
        ]._layer_node.get_node(3)

        texture = volume._texture

        new_texture_data = np.asarray(
            data,
            dtype=dtype,
        )

        # Note: due to odd dimensions in scale pyramids sometimes we have off by 1
        ntd_slice = (
            slice(0, layer.data[texture_slice].shape[1]),
            slice(0, layer.data[texture_slice].shape[2]),
            slice(0, layer.data[texture_slice].shape[3]),
        )
        if len(new_texture_data.shape) == 3:
            import pdb

            pdb.set_trace()

        layer.data[texture_slice] = new_texture_data[
            : layer.data[texture_slice].shape[0],
            : layer.data[texture_slice].shape[1],
            : layer.data[texture_slice].shape[2],
            : layer.data[texture_slice].shape[3],
        ]

        """
    dims.current_step
(266, 1, 494, 64, 67)
(Pdb) array.shape
(532, 2, 988, 256, 271)
(Pdb)
"""

        # TODO explore efficiency of either approach, or maybe even an alternative
        # Writing a texture with an offset is slower
        # texture.set_data(new_texture_data, offset=texture_offset)
        texture.set_data(np.asarray(layer.data[texture_slice]).squeeze())
        # TODO we might need to zero out parts of the texture when there is a ragged boundary size

        volume.update()

        # Translate the layer we're rendering to the right place
        # TODO: this is called too many times. we only need to call it once for each time it changes
        # Note: this can trigger a refresh, we should do it after setting data to not trigger an
        #       extra materialization with
        # TODO this might be a race condition with multiple on_yielded events
        if node_offset is not None:
            try:
                # Trigger a refresh of our layer
                layer.refresh()

                next_layer_name = f"{container}/{dataset}/s{scale - 1}"
                next_layer = viewer.layers[next_layer_name]
                next_layer.translate = node_offset
                LOGGER.debug(
                    f" moving next layer scale {scale - 1} to {node_offset}"
                )
            except Exception as e:
                import pdb

                pdb.traceback.print_exception(e)
                pdb.traceback.print_stack()

                pdb.set_trace()
            """
    This error is triggered here infrequently.
            
    
    WARNING: Traceback (most recent call last):
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/toolz/functoolz.py", line 304, in __call__
    return self._partial(*args, **kwargs)
  File "/Users/kharrington/nesoi/repos/napari/examples/are_the_chunks_in_view.py", line 661, in update_chunk
    layer.translate = node_offset
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/base/base.py", line 643, in translate
    self._clear_extent()
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/base/base.py", line 806, in _clear_extent
    self.refresh()
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/base/base.py", line 1212, in refresh
    self.set_view_slice()
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/base/base.py", line 974, in set_view_slice
    self._set_view_slice()
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/image.py", line 735, in _set_view_slice
    self._update_slice_response(response)
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/image.py", line 802, in _update_slice_response
    self._load_slice(slice_data)
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/image.py", line 830, in _load_slice
    if self._slice.load(data):
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/_image_slice.py", line 125, in load
    return self.loader.load(data)
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/_image_loader.py", line 22, in load
    data.load_sync()
  File "/Users/kharrington/git/kephale/nesoi/repos/napari/napari/layers/image/_image_slice_data.py", line 44, in load_sync
    self.image = np.asarray(self.image)
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/array/core.py", line 1699, in __array__
    x = self.compute()
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/base.py", line 314, in compute
    (result,) = compute(self, traverse=False, **kwargs)
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/base.py", line 599, in compute
    results = schedule(dsk, keys, **kwargs)
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/threaded.py", line 89, in get
    results = get_async(
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/local.py", line 516, in get_async
    f(key, res, dsk, state, worker_id)
  File "/Users/kharrington/mambaforge/envs/nesoi/lib/python3.10/site-packages/dask/cache.py", line 56, in _posttask
    duration = default_timer() - self.starttimes[key]
KeyError: ('setitem-0894a5557d8b18e0f8a3165b7ad0b979', 0, 0, 0, 0)"""

        # The node translate calls made when zeroing will update everyone except 0
        if scale == 0:
            layer.refresh()

        toc = time.perf_counter()

        LOGGER.debug(
            f"update_chunk {scale} {array_offset} at node offset {node_offset} with size {new_texture_data.shape} took {toc - tic:0.4f} seconds"
        )


@tz.curry
def add_subnodes(
    event,
    view_slice,
    scale=0,
    viewer=None,
    cache_manager=None,
    arrays=None,
    chunk_maps=None,
    container="",
    dataset="",
    alpha=0.8,
    scale_factors=[],
    worker_map={},
    viewer_lock=None,
):
    """Recursively add multiscale chunks to a napari viewer for some multiscale arrays

    Note: scale levels are assumed to be 2x factors of each other

    Parameters
    ----------
    view_slice : tuple or list of slices
        A tuple/list of slices defining the region to display
    scale : float
        The scale level to display. 0 is highest resolution
    viewer : viewer
        a napari viewer that the nodes will be added to
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
    """

    camera = viewer.camera.copy()

    # TODO hardcoded dtype
    dtype = np.uint16

    if "worker" in worker_map:
        worker_map["worker"].quit()

    # TODO slice the arrays into 3D now

    arrays_3d = []
    for array in arrays:
        # TODO This assumes a 3D ordering
        slice_to_3d = tuple(
            [slice(el, el + 1) for el in viewer.dims.current_step[:-3]]
            + [slice(0, max_size) for max_size in array.shape[-3:]]
        )
        arrays_3d.append(array[slice_to_3d])

    worker_map["worker"] = render_sequence_caller(
        view_slice,
        scale,
        camera,
        cache_manager,
        arrays=arrays,
        chunk_maps=chunk_maps,
        container=container,
        dataset=dataset,
        scale_factors=scale_factors,
        alpha=alpha,
        dtype=dtype,
        dims=viewer.dims.copy(),
    )

    # TODO keep track of a set of keys that describe each chunk that is already rendered

    worker_map["worker"].yielded.connect(
        update_chunk(
            viewer_lock=viewer_lock,
            viewer=viewer,
            container=container,
            dataset=dataset,
            dtype=dtype,
        )
    )
    worker_map["worker"].start()


@magic_factory(
    call_button="Poor Octree Renderer",
)
def poor_octree_widget(
    viewer: "napari.viewer.Viewer",
    alpha: float = 0.8,
    colormap: str = "multiscale",
):
    # TODO get this working with a non-remote large data sample
    # Chunked, multiscale data

    # These datasets have worked at one point in time
    # large_image = openorganelle_mouse_kidney_labels()
    # large_image = idr0044A()
    large_image = luethi_zenodo_7144919()
    # large_image = idr0051A()

    # These datasets need testing
    # large_image = idr0075A()
    # large_image = openorganelle_mouse_kidney_em()
    # TODO there is a problem with datasets that for some reason have shape == chunksize
    #      these datasets overflow because of memory issues

    # view_interval = ((0, 0, 0), [3 * el for el in chunk_strides[3]])
    # view_interval = ((0, 0, 0), (6144, 2048, 4096))

    cache_manager = ChunkCacheManager(cache_size=6e9)

    # TODO if the lowest scale level of these arrays still exceeds texture memory, this breaks
    multiscale_arrays = large_image["arrays"]

    # Testing with ones is pretty useful for debugging chunk placement for different scales
    # TODO notice that we're using a ones array for testing instead of real data
    # multiscale_arrays = [da.ones_like(array) for array in multiscale_arrays]

    multiscale_chunk_maps = [
        chunk_centers(array)
        for scale_level, array in enumerate(multiscale_arrays)
    ]

    multiscale_grids = [
        np.array(list(chunk_map)) for chunk_map in multiscale_chunk_maps
    ]

    # view_interval = ((0, 0, 0), multiscale_arrays[0].shape)
    view_slice = [
        slice(0, multiscale_arrays[-1].shape[idx])
        for idx in range(len(multiscale_arrays[-1].shape))
    ]
    # Forcing 3D here
    view_slice = view_slice[-3:]

    # viewer = napari.Viewer(ndisplay=3)

    if colormap == "multiscale":
        colormaps = {
            0: "red",
            1: "blue",
            2: "green",
            3: "yellow",
            4: "bop purple",
        }
    else:
        colormaps = {0: "gray", 1: "gray", 2: "gray", 3: "gray", 4: "gray"}

    # Initialize layers
    container = large_image["container"]
    dataset = large_image["dataset"]
    scale_factors = large_image["scale_factors"]

    # Initialize worker
    worker_map = {}

    # from napari.layers.image.image import Image
    # Image._set_view_slice = lambda x: None

    scale = len(multiscale_arrays) - 1
    viewer.add_image(
        da.ones_like(multiscale_arrays[scale], dtype=np.uint16),
        blending="additive",
        scale=scale_factors[scale],
        colormap=colormaps[scale],
        opacity=0.8,
        rendering="mip",
        name=f"{container}/{dataset}/s{scale}",
        contrast_limits=[0, 500],
    )
    for scale in range(len(multiscale_arrays) - 1):
        relative_scale_factor = [
            this_scale / next_scale
            for this_scale, next_scale in zip(
                scale_factors[scale], scale_factors[scale - 1]
            )
        ]

        # TODO Make sure this is still smaller than the array
        scale_shape = np.array(multiscale_arrays[scale + 1].chunksize) * 2

        # TODO this is really not the right thing to do, it is an overallocation
        scale_shape[:-3] = multiscale_arrays[scale + 1].shape[:-3]

        viewer.add_image(
            da.ones(
                scale_shape,
                dtype=np.uint16,
            ),
            blending="additive",
            scale=scale_factors[scale],
            colormap=colormaps[scale],
            opacity=0.8,
            rendering="mip",
            name=f"{container}/{dataset}/s{scale}",
            contrast_limits=[0, 500],
        )

    viewer_lock = threading.Lock()

    viewer.dims.current_step = (0, 5, 135, 160)

    # Hooks and calls to start rendering
    @viewer.bind_key("k")
    def camera_response(event):
        add_subnodes(
            event,
            view_slice,
            scale=len(multiscale_arrays) - 1,
            viewer=viewer,
            cache_manager=cache_manager,
            arrays=multiscale_arrays,
            chunk_maps=multiscale_chunk_maps,
            container=large_image["container"],
            dataset=large_image["dataset"],
            scale_factors=scale_factors,
            worker_map=worker_map,
            viewer_lock=viewer_lock,
            alpha=alpha,
        )

    # Trigger the first render pass
    camera_response(None)

    # TODO note that debounced uses threading.Timer
    # Connect to camera
    viewer.camera.events.connect(
        debounced(
            ensure_main_thread(camera_response),
            timeout=1000,
        )
    )

    # Connect to dims (to get sliders)
    viewer.dims.events.connect(
        debounced(
            ensure_main_thread(camera_response),
            timeout=1000,
        )
    )

    # napari.run()

    # Overall TODO 2023-02-15
    # TODO refactor/cleanup
    # TODO check with timeseries, if not then disable sliders


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()

    widget = poor_octree_widget()

    # widget(viewer)

    viewer.window.add_dock_widget(widget, name="Poor Octree Renderer")
    # widget_demo.show()
