import numpy as np
import pandas as pd
import napari

from scipy.spatial.transform import Rotation as R

grid_coord = np.linspace(0, 500, num=10)
grid = np.stack(
    np.meshgrid(*[grid_coord] * 3),
    axis=-1,
).reshape((-1, 3))

viewer = napari.Viewer(ndisplay=3)


def rotation_matrix_from_camera(camera: napari.components.Camera) -> np.ndarray:
    return R.from_euler(seq='yzx', angles=camera.angles, degrees=True)

def distance_from_camera_centre_line(points, camera):
    view_direction = camera.view_direction
    dot_products = points @ view_direction
    projected = view_direction * np.reshape(dot_products, (-1, 1))
    distances = np.linalg.norm(projected - points, axis=-1)
    return distances


initial_dist = distance_from_camera_centre_line(grid, viewer.camera)

viewer.add_points(grid, features={'distance': initial_dist}, face_color='distance')

napari.run()
