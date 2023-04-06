import numpy as np

import napari

viewer = napari.Viewer()
viewer.axes.visible = True

print(f'{viewer.axis_labels=}')  # -> ()

image = viewer.add_image(
    np.ones((5, 3, 2)),
    colormap='red',
)
print(f'{viewer.axis_labels=}')  # -> ("-3", "-2", "-1")

image = viewer.add_image(
    np.ones((3, 2)),
    colormap='green',
)
print(f'{viewer.axis_labels=}')  # -> ("-2", "-1")

image = viewer.add_image(
    np.ones((6, 4, 3, 2)),
    colormap='blue',
)
print(f'{viewer.axis_labels=}')  # -> ("-4", "-3", "-2", "-1")

#napari.run()
