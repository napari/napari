"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # add the image
    viewer = napari.view_image(rgb2gray(data.astronaut()))
    # add the points
    points = np.array([[100, 100], [200, 200], [333, 111]])
    size = np.array([10, 20, 20])
    annotations = {
        'point_class': np.array(['A', 'A', 'B']),
        'good_point': np.array([True, False, False])
    }
    face_color_cycle = ['blue', 'green']
    points_layer = viewer.add_points(
        points,
        size=size,
        annotations=annotations,
        face_color='point_class',
        face_color_cycle=face_color_cycle
    )

    # change the annotation that sets the face_color
    points_layer.face_color = 'good_point'

    # bind a function to toggle the good_point annotation of the selected points
    @viewer.bind_key('t')
    def toggle_point_annotation(viewer):
        selected_points = viewer.layers[1].selected_data
        if selected_points:
            selected_annotations = viewer.layers[1].annotations['good_point'][selected_points]
            toggled_annotations = np.logical_not(selected_annotations)
            viewer.layers[1].annotations['good_point'][selected_points] = toggled_annotations

            # we need to manually refresh since we did not use the Points.annotations setter
            points_layer._refresh_face_color()
