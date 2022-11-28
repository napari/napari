import os
import numpy as np
import napari
import napari.viewer


"""
This example generates many random shapes.

There currently is a bug in triangulation that requires this additional step of sanitizing the shape data: https://github.com/orgs/napari/projects/18/views/2
"""


# logging.getLogger().setLevel(0)


def generate_shapes(filename):
    # image_data = np.squeeze(cells3d()[:, 1, :, :])
    # delayed_image_data = DelayedArray(image_data, delay_s=1)

    # From https://github.com/napari/napari/blob/main/examples/nD_shapes.py
    # create one random polygon per "plane"

    shapes_per_slice = 1000

    all_shapes = None

    np.random.seed(0)
    for k in range(shapes_per_slice):

        planes = np.tile(np.arange(128).reshape((128, 1, 1)), (1, 5, 1))
        corners = np.random.uniform(0, 128, size=(128, 5, 2))
        shapes = np.concatenate((planes, corners), axis=2)

        if all_shapes is not None:
            all_shapes = np.concatenate((all_shapes, shapes), axis=0)
        else:
            all_shapes = shapes

    print('all_shapes', all_shapes.shape)

    from vispy.geometry.polygon import PolygonData

    good_shapes = []

    for shape in all_shapes:

        # Use try/except to filter all bad shapes
        try:
            vertices, triangles = PolygonData(
                vertices=shape[:, 1:]
            ).triangulate()
        except:
            pass
        else:
            good_shapes.append(shape)

    print(len(good_shapes))
    np.savez(filename, shapes=good_shapes)


test_filename = '/tmp/napari_example_shapes.npz'

# Create the example shapes if they do not exist
if not os.path.exists(test_filename):
    print(
        'Shapes file does not exist yet. Generating shapes. This may take a couple of minutes...'
    )
    generate_shapes(test_filename)

# Load the shapes
with np.load(test_filename) as data:
    shapes = data['shapes']

# Test shapes in viewer
viewer = napari.Viewer()
viewer.show()

shapes_layer = viewer.add_shapes(
    np.array(shapes),
    shape_type='polygon',
    name='sliced',
)

napari.run()
