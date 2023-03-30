import numpy as np

import napari

viewer = napari.Viewer()
viewer.axes.visible = True

print(f'{viewer.axis_labels=}')  # -> []

image = viewer.add_image(
    np.ones((5, 3, 2)),
    axis_labels=("time", "y", "x"),
    colormap='red',
)
print(f'{viewer.axis_labels=}')  # -> []
sliced_data = image._make_slice_request()().data

image = viewer.add_image(
    np.ones((4, 3, 2)),
    axis_labels=("z", "y", "x"),
    colormap='green',
)
print(f'{viewer.axis_labels=}')  # -> ["z", "time", "y", "x"]

image = viewer.add_image(
    np.ones((6, 4, 3, 2)),
    axis_labels=["freq", "z", "y", "x"],
    colormap='blue',
)
print(f'{viewer.axis_labels=}')  # -> ["freq", "z", "time", "y", "x"]

#image = viewer.add_image(4 * np.ones((3, 2)), axis_labels=None)
#image.axis_labels # -> ["y", "x"]
#viewer.dims.ndim # -> 5
#viewer.dims.axis_labels # -> ["freq", "z", "time", "y", "x"]

#napari.run()
