import itertools
import dask.array as da
from skimage import data
import numpy as np
import pandas as pd
import napari
import toolz as tz
from psygnal import debounced

from scipy.spatial.transform import Rotation as R


def chunk_centers(array : da.Array):
    """Return a dictionary mapping chunk centers to chunk slices."""
    start_pos = [np.cumsum(sizes) - sizes for sizes in array.chunks]
    middle_pos = [
        np.cumsum(sizes) - (np.array(sizes)/2) for sizes in nuclei_dask.chunks
    ]
    end_pos = [np.cumsum(sizes) for sizes in nuclei_dask.chunks]
    all_start_pos = list(itertools.product(*start_pos))
    all_middle_pos = list(itertools.product(*middle_pos))
    all_end_pos = list(itertools.product(*end_pos))
    chunk_slices = []
    for start, end in zip(all_start_pos, all_end_pos):
        chunk_slice = [
            slice(start_i, end_i) for start_i, end_i in zip(start, end)
        ]
        chunk_slices.append(tuple(chunk_slice))

    mapping = dict(zip(all_middle_pos, chunk_slices))
    return mapping


def rotation_matrix_from_camera(camera: napari.components.Camera) -> np.ndarray:
    return R.from_euler(seq='yzx', angles=camera.angles, degrees=True)


def visual_depth(points, camera):
    """Compute visual depth from camera position to a point or array of points. """
    view_direction = camera.view_direction
    points_relative_to_camera = points - camera.center
    projected_length = points_relative_to_camera @ view_direction
    return projected_length


def distance_from_camera_centre_line(points, camera):
    """Compute distance from a point or array of points to camera center line.

    This is the line aligned to the camera view direction and passing through
    the camera's center point, aka camera.position.
    """
    view_direction = camera.view_direction
    projected_length = visual_depth(points, camera)
    projected = view_direction * np.reshape(projected_length, (-1, 1))
    points_relative_to_camera = points - camera.center  # for performance, don't compute this twice in both functions 
    distances = np.linalg.norm(projected - points_relative_to_camera, axis=-1)
    return distances


def prioritised_chunk_loading(depth, distance, zoom, alpha=1.):
    chunk_load_priority = depth + alpha * zoom * distance
    return chunk_load_priority


@tz.curry
def update_point_colors(event, viewer, alpha=1.):
    """Update the points based on their distance to current camera.

    Parameters:
    -----------
    viewer : napari.Viewer
        Current viewer
    event : camera.events.angles event
        The event triggered by changing the camera angles
    """
    points_layer = viewer.layers['grid']
    points = points_layer.data
    distances = distance_from_camera_centre_line(points, viewer.camera)
    depth = visual_depth(points, viewer.camera)
    priorities = prioritised_chunk_loading(
        depth, distances, viewer.camera.zoom, alpha=alpha
    )
    points_layer.features = pd.DataFrame({
        'distance': distances, 'depth': depth, 'priority': priorities
    })
    points_layer.face_color = 'priority'
    points_layer.refresh()


@tz.curry
def update_shown_chunk(event, viewer, chunk_map, array, alpha=1.):
    points = np.array(list(centers.keys()))
    distances = distance_from_camera_centre_line(points, viewer.camera)
    depth = visual_depth(points, viewer.camera)
    priorities = prioritised_chunk_loading(
        depth, distances, viewer.camera.zoom, alpha=alpha
    )
    first_priority_idx = np.argmin(priorities)
    first_priority_coord = tuple(points[first_priority_idx])
    chunk_slice = chunk_map[first_priority_coord]
    offset = [sl.start for sl in chunk_slice]
    hi_res_layer = viewer.layers['high-res']
    hi_res_layer.data = array[chunk_slice]
    hi_res_layer.translate = offset
    hi_res_layer.refresh()


if __name__ == '__main__':

    # Chunked, multiscale data
    cells = data.cells3d()
    nuclei = cells[:, 1]
    nuclei_dask = da.from_array(nuclei, chunks=(20, 64, 64))
    nuclei_down = nuclei_dask[::2, ::2, ::2]
    multiscale_nuclei = [nuclei_dask, nuclei_down]

    centers = chunk_centers(nuclei_dask)
    grid = np.array(list(centers.keys()))

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(multiscale_nuclei)

    initial_dist = distance_from_camera_centre_line(grid, viewer.camera)
    initial_depth = visual_depth(grid, viewer.camera)
    initial_priority = prioritised_chunk_loading(
        initial_depth, initial_dist, viewer.camera.zoom, alpha=1.0
    )
    features = pd.DataFrame({
        'distance': initial_dist,
        'depth': initial_depth,
        'priority': initial_priority,
    })

    viewer.add_points(
        grid,
        features={
            'distance': initial_dist,
            'depth': initial_depth,
            'priority': initial_priority
        },
        face_color='priority',
    )
    viewer.camera.events.connect(
        debounced(
            update_point_colors(viewer=viewer, alpha=1.0),
            timeout=100,
        )
    )
    napari.run()
