"""
Add points on nD shapes
=======================

Add points on nD shapes in 3D using a mouse callback

.. tags:: visualization-nD
"""

import numpy as np

import napari

# Create rectangles in 4D
data = [
        [
            [0, 50, 75, 75],
            [0, 50, 125, 75],
            [0, 100, 125, 125],
            [0, 100, 75, 125]
        ],
        [
            [0, 10, 75, 75],
            [0, 10, 125, 75],
            [0, 40, 125, 125],
            [0, 40, 75, 125]
        ],
        [
            [1, 100, 75, 75],
            [1, 100, 125, 75],
            [1, 50, 125, 125],
            [1, 50, 75, 125]
        ]
    ]

shapes_data = np.array(data)

# add an empty 4d points layer
viewer = napari.view_points(ndim=4, size=3)
points_layer = viewer.layers[0]

# add the shapes layer to the viewer
features = {'index': [0, 1, 2]}
for shape_type, mult in {('ellipse', 1), ('rectangle', -1)}:
    shapes_layer = viewer.add_shapes(
        shapes_data * mult,
        face_color=['magenta', 'green', 'blue'],
        edge_color='white',
        blending='additive',
        features=features,
        text='index',
        shape_type=shape_type,
    )

    @shapes_layer.mouse_drag_callbacks.append
    def on_click(layer, event):

        shape_index, intersection_point = layer.get_index_and_intersection(
            event.position, event.view_direction, event.dims_displayed
        )

        if (shape_index is not None) and (intersection_point is not None):
            points_layer.add(intersection_point)


for d in data:
    viewer.add_points(np.array(d))
# set the viewer to 3D rendering mode with the first two rectangles in view
viewer.dims.ndisplay = 3
viewer.dims.set_point(axis=0, value=0)
viewer.camera.angles = (70, 30, 150)
viewer.camera.zoom = 2.5

if __name__ == '__main__':
    napari.run()
