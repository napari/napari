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

    # create properties for each point
    properties = {
        'confidence': np.array([1, 0.5, 0]),
        'good_point': np.array([True, False, False])
    }

    # define the color cycle for the face_color annotation
    face_color_cycle = ['blue', 'green']

    # create a points layer where the face_color is set by the good_point property
    # and the edge_color is set via a color map (grayscale) on the confidence property.
    points_layer = viewer.add_points(
        points,
        properties=properties,
        size=20,
        edge_width=7,
        edge_color='confidence',
        edge_colormap='gray',
        face_color='good_point',
        face_color_cycle=face_color_cycle
    )

    # set the edge_color mode to colormap
    points_layer.edge_color_mode = 'colormap'

    # bind a function to toggle the good_point annotation of the selected points
    @viewer.bind_key('t')
    def toggle_point_annotation(viewer):
        selected_points = list(points_layer.selected_data)
        if len(selected_points) > 0:
            props = points_layer.properties
            good_point = props['good_point']
            good_point[list(selected_points)] = ~good_point[list(selected_points)]
            props['good_point'] = good_point
            points_layer.properties = props