import numpy as np
import pandas as pd
import napari
import toolz as tz
from psygnal import debounced

from scipy.spatial.transform import Rotation as R


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


if __name__ == '__main__':
    # Just make a grid of points that we can colour as we like
    grid_coord = np.linspace(0, 500, num=10)
    grid = np.stack(
        np.meshgrid(*[grid_coord] * 3),
        axis=-1,
    ).reshape((-1, 3))

    viewer = napari.Viewer(ndisplay=3)

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
        debounced(update_point_colors(viewer=viewer, alpha=1.0),
        timeout=100)
    )
    napari.run()
