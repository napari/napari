"""
Layer Text Scaling
==================

Display points layer with text features.
By using the `scaling` property of the text layer, the text can be scaled as canvas zoom changes.
This scaling is useful when there are many points and the text would otherwise overlap when zoomed out, or become illegible when zoomed in.

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
    'size': 10,
    'color': 'magenta',
    'anchor': 'center',
    'scaling': True
}
points_layer = viewer.add_points(
    points,
    size=20,
    text=text_scaled,
    name='scaled_text'
)
# points_layer.text.scaling = True  # alternative way to set scaling

# add a second points layer slightly offset and show without scaling
text_unscaled = {
    'string': ['Point 1', 'Point 2', 'Point 3', 'Point 4'],
    'size': 10,
    'color': 'green',
    'anchor': 'center',
}
points_layer2 = viewer.add_points(
    points + [40, 0],
    size=20,
    text=text_unscaled,
    name='unscaled_text'
)

for layer in viewer.layers:
    layer.scale = (3,3) # change to (1,1) to see how things look different

points_layer.events.scale_factor.connect(lambda: print(f"Scale factor change: {points_layer.scale_factor}"))
points_layer.events.scale_factor.connect(lambda: print(f"Scale factor change, zoom: {viewer.camera.zoom}"))
viewer.camera.events.zoom.connect(lambda: print(f"Zoom change: {viewer.camera.zoom}"))

if __name__ == '__main__':
    napari.run()
