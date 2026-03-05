"""
Features table widget
======================

Visualize and interact with the `features` of a layer via the builtin
Features table widget.
The widget can be used to navigate, edit, select, copy/paste, and save the features
of a compatible layer (such as Points, Labels or Shapes).
Selection is synchronized between the table widget and the data. When selecting rows in
the table widget, the corresponding data will be selected on the layer. When selecting
data on the layer, the corresponding row in the table widget will be selected.
on the layer and viceversa.

If multiple layers are selected, their features will be joined and displayed together in the table.
The layer name column indicates the origin layer of each feature row and a "shared columns" toggle
allows to switch between showing only the columns shared by all selected layers or all columns.

.. tags:: gui, features-table
"""

import numpy as np
import pandas as pd

import napari

viewer = napari.Viewer(ndisplay=3)

features = pd.DataFrame({
    'a': np.random.rand(10),
    'b': ['stuff'] * 10,
    'c': np.random.choice([0, 1], 10).astype(bool),
    'd': pd.Series(np.random.choice(['x', 'y', 'z'], 10), dtype='category')
})
features.loc[3, 'a'] = np.nan
features.loc[2, 'b'] = 'asd'

features2 = pd.DataFrame({
    'a': np.random.rand(10) + 2,
    'f': np.random.randint(0, 100, 10),
})
viewer.add_image(np.random.rand(10, 10))

points = np.random.rand(10, 3) * 10
faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
                   [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])
surface_layer = viewer.add_surface((points, faces, features2['a'].values), features=features2, colormap='plasma')
points_layer = viewer.add_points(points, size=1, features=features, name='Points')

viewer.layers.selection.add(points_layer)
viewer.layers.selection.add(surface_layer)
viewer.window.add_plugin_dock_widget('napari', 'Features table widget')

if __name__ == '__main__':
    napari.run()
