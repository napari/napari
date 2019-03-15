"""
Display one markers layer ontop of one image layer using the add_markers and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = Viewer()
    window = Window(viewer)
    # add the image
    viewer.add_image(data.camera())
    viewer.layers[0].visible = False

    # add the shapes
    ellipses = np.array([[[195, 180], [ 45, 130]], [[295,  80], [395, 380]], [[ 95,
                        80], [105,  90]]])
    points = np.array([[100, 100], [200, 200], [333, 111]])
    polygons = [points+[-22,101], points-[140, -33], points+[19,-21]]

    #viewer.add_shapes(polygons=polygons, ellipses=ellipses)

    shapes = [{'shape_type': 'polygon', 'data': p, 'face_color': 'coral'} for p in polygons]

    viewer.add_shapes(shapes)
    #
    # # primary path
    # # if ShapeList or list ->
    # viewer.add_shapes(ShapeList(Shape, Shape, Shape, ...))
    # viewer.add_shapes([Shape, Shape, Shape, ...])
    #
    # # convienience
    # # if dict ->
    # viewer.add_shapes(type='rectangle', data=np.array([[10, 10], [10, 10]]))
    # viewer.add_shapes(type=['rectangle','polygon'], data=[np.array([[10,10],[10,10]]), np.array([10,20,30])])
    #
    # # converters (to be added down the road)
    # viewer.add_shapes_from_geojson({...})
    # viewer.add_shapes_from_shapely(...)
    #
    # viewer.add_shapes(shape_type='rectangle', data=np.array([[10, 10], [10, 10]]), 'color'='red')


    # viewer.layers[-1].edge_width = 10
    # viewer.layers[-1].edge_color = 'coral'
