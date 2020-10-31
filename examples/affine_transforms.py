"""
Display an image and its corners before and after an affine transform
"""
import numpy as np
import napari

# Create a random image
image = np.random.random((5, 5))

# Define an affine transform
affine = np.array([[1, -1, 4], [2, 3, 2], [0, 0, 1]])

# Define the corners of the image, including in homogeneous space
corners = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
corners_h = np.concatenate([corners, np.ones((4, 1))], axis=1)

with napari.gui_qt():
    viewer = napari.Viewer()

    # Add the original image and its corners
    viewer.add_image(image, name='background', colormap='red', opacity=.5)
    viewer.add_points(corners_h[:, :-1], size=0.5, opacity=.5, face_color=[0.8, 0, 0, 0.8], name='bg corners')

    # Add another copy of the image, now with a transform, and add its transformed corners
    viewer.add_image(image, colormap='blue', opacity=.5, name='moving', affine=affine)
    viewer.add_points((corners_h @ affine.T)[:, :-1], size=0.5, opacity=.5, face_color=[0, 0, 0.8, 0.8], name='mv corners')

    # Reset the view
    viewer.reset_view()

    # Note how the transformed corner points remain at the corners of the transformed image