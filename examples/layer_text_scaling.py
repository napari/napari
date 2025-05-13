"""
Layer Text Scaling
==================

Display points layer with text features.
By using the `scaling` property of the text layer, the text is scaled with the layers as canvas zoom changes.
Scaling can also be thought of as fixing the height of the text in world coordinates,
whereas no scaling fixes the height of the text in screen / canvas pixels.
This world scaling is useful when there are many points and the text would otherwise overlap when zoomed out, or become illegible when zoomed in.

.. tags:: visualization-advanced
"""

import numpy as np

import napari

viewer = napari.Viewer()
viewer.open_sample('napari', 'astronaut')

# add points with text features
points = np.array([[100, 100], [120, 120], [140, 140], [160, 160]])
text_scaled = {
    'string': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],
    'size': 8,
    'color': 'magenta',
    'anchor': 'center',
    'scaling': True
}
points_layer = viewer.add_points(
    points,
    size=20,
    text=text_scaled,
    name='scaled_text (magenta)'
)
# points_layer.text.scaling = True  # alternative way to set scaling

# add a second points layer slightly offset and show without scaling
text_unscaled = {
    'string': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],
    'size': 8,
    'color': 'green',
    'anchor': 'center',
}
points_layer2 = viewer.add_points(
    points + [40, 0],
    size=20,
    text=text_unscaled,
    name='unscaled_text (green)'
)

shape_text = {
    'string': ['Triangle'],
    'size': 12,
    'scaling': True,
}
shapes_layer = viewer.add_shapes(
    data=[[100, 300], [200, 300], [100, 400]],
    shape_type=['polygon'],
    text=shape_text,
    name='scaled_text_shapes'
)

for layer in viewer.layers:
    layer.scale = (3, 3)

viewer.reset_view()

if __name__ == '__main__':
    napari.run()
