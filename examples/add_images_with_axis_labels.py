import numpy as np

from napari.components import ViewerModel

viewer = ViewerModel()
viewer.axes.visible = True

print(f'{viewer.axis_labels=}')  # -> ()

image = viewer.add_image(np.ones((5, 3, 2)), axis_labels=("time", "y", "x"))
print(f'{viewer.axis_labels=}')  # -> ("time", "y", "x")

image = viewer.add_image(np.ones((4, 3, 2)), axis_labels=("z", "y", "x"))
print(f'{viewer.axis_labels=}')  # -> ("z", "time", "y", "x")

data, axes = image.get_slice({"z": 2, "y": slice(None), "x": slice(None)})
print(f'{data.shape=}\n{axes=}')  # -> (3, 2), ("y", "x")

data, axes = image.get_slice({"z": slice(None), "y": 1, "x": slice(None)})
print(f'{data.shape=}\n{axes=}')  # -> (4, 2), ("z", "x")

data, axes = image.get_slice({"x": slice(None), "y": 1, "z": slice(None)})
print(f'{data.shape=}\n{axes=}')  # -> (2, 4), ("x", "z")

image = viewer.add_image(np.ones((6, 4, 3, 2)), axis_labels=("freq", "z", "y", "x"))
print(f'{viewer.axis_labels=}')  # -> ("freq", "z", "time", "y", "x")

data, axes = image.get_slice({"z": slice(None), "y": 1, "x": slice(None)})
print(f'{data.shape=}\n{axes=}')  # -> (6, 2, 4), ("freq", "x", "z")
