"""
Cursor ray
==========

Depict a ray through a layer in 3D to demonstrate interactive 3D functionality

.. tags:: interactivity
"""
import numpy as np
import napari

sidelength_data = 64
n_points = 10

# data to depict an empty volume, its bounding box and points along a ray
# through the volume
volume = np.zeros(shape=(sidelength_data, sidelength_data, sidelength_data))
bounding_box = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
) * sidelength_data
points = np.zeros(shape=(n_points, 3))

# point sizes
point_sizes = np.linspace(0.5, 2, n_points, endpoint=True)

# point colors
green = [0, 1, 0, 1]
magenta = [1, 0, 1, 1]
point_colors = np.linspace(green, magenta, n_points, endpoint=True)

# create viewer and add layers for each piece of data
viewer = napari.Viewer(ndisplay=3)
bounding_box_layer = viewer.add_points(
    bounding_box, face_color='cornflowerblue', name='bounding box'
)
ray_layer = viewer.add_points(
    points, face_color=point_colors, size=point_sizes, name='cursor ray'
)
volume_layer = viewer.add_image(volume, blending='additive')


# callback function, called on mouse click when volume layer is active
@volume_layer.mouse_drag_callbacks.append
def on_click(layer, event):
    near_point, far_point = layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed
    )
    if (near_point is not None) and (far_point is not None):
        ray_points = np.linspace(near_point, far_point, n_points, endpoint=True)
        if ray_points.shape[1] != 0:
            ray_layer.data = ray_points


if __name__ == '__main__':
    napari.run()
