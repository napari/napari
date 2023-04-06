import numpy as np

from napari.components import ViewerModel

viewer = ViewerModel()
viewer.axes.visible = True

print(f'{viewer.axis_labels=}')  # -> ()

image = viewer.add_image(np.ones((5, 3, 2)))
print(f'{viewer.axis_labels=}')  # -> ("-3", "-2", "-1")

image = viewer.add_image(np.ones((3, 2)))
print(f'{viewer.axis_labels=}')  # -> ("-2", "-1")

image = viewer.add_image(np.ones((6, 4, 3, 2)))
print(f'{viewer.axis_labels=}')  # -> ("-4", "-3", "-2", "-1")
