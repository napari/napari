import napari
import numpy as np
import dask.array as da
from skimage import data


image4d = da.random.random(
        (4000, 32, 256, 256),
        chunks=(1, 32, 256, 256),
        )
pts_coordinates = np.random.random((50000, 3)) * image4d.shape[1:]
pts_values = da.random.random((50000, 4000), chunks=(50000, 1))

viewer = napari.Viewer(ndisplay=3)
image_layer = viewer.add_image(
        image4d, opacity=0.5
        )
pts_layer = viewer.add_points(
        pts_coordinates,
        properties={'value': np.asarray(pts_values[:, 0])},
        face_color='value',
        size=2,
        )


def set_pts_properties(pts_layer, values_table, step):
    # step is a 4D coordinate with the current slider position for each dim
    column = step[0]  # grab the leading ("time") coordinate
    pts_layer.properties['value'] = np.asarray(values_table[:, column])
    pts_layer.face_color = 'value'  # force properties refresh


viewer.dims.events.current_step.connect(
        lambda event: set_pts_properties(pts_layer, pts_values, event.value)
        )


napari.run()
