"""
This example demonstrates how to export point and shape coordinates to csv.
"""

import numpy as np
import pandas as pd
from skimage import data

import napari
from napari.layers import Points, Shapes

with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer()
    # Add an image to the viewer
    image_layer = viewer.add_image(data.camera(), name='photographer')

    # POINT COORDINATES
    data_for_points = np.array([
        [160, 160],
        [210, 100],
        [280, 170],
        [340, 120],
        [410, 150]]
    )
    points = viewer.add_points(data_for_points)
    points.selected_data = [1, 2]  # select the second and third point
    # Save points coordinates to a csv file
    points_table, points_column_names = points.to_table('points.csv')
    # Optionally, you can save only the currently selected points
    selected_points_table, selected_points_cols = points.to_table(
        'selected_points.csv', selected_only=True
    )

    # SHAPES COORDINATES
    rectangle_data = np.array([
        [241, 426],
        [415, 426],
        [415, 487],
        [241, 487],
    ])
    ellipse_data = np.array([
        [ 51, 346],
        [129, 346],
        [129, 478],
        [ 51, 478],
    ])
    shapes = viewer.add_shapes([rectangle_data, ellipse_data],
                               shape_type=['rectangle', 'ellipse']
    )
    shapes.selected_data = [1]  # select the ellipse
    # Save shapes coordinates to a csv file
    shapes_table, shapes_column_names = shapes.to_table('shapes.csv')
    # Optionally, you can save only the currently selected shapes
    selected_shapes_table, selected_shapes_cols = shapes.to_table(
        'selected_shapes.csv', selected_only=True
    )

    # PANDAS DATAFRAMES
    # You may like to have this data as a pandas dataframe for further analysis
    df_points = pd.DataFrame(points_table, columns=points_column_names)
    df_shapes = pd.DataFrame(shapes_table, columns=shapes_column_names)
    print('Points dataframe:')
    print(df_points)
    print('Shapes dataframe:')
    print(df_shapes)
    # Use `index=None` when saving pandas dataframes if you need the output to
    # match the csv output using the napari to_table() method.
    df_points.to_csv('points_from_dataframe.csv', index=None)
    df_shapes.to_csv('shapes_from_dataframe.csv', index=None)
