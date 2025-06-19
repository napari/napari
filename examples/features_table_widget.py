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
viewer.add_image(np.random.rand(10, 10))

points = viewer.add_points(np.random.rand(10, 3) * 10, size=1, features=features)

viewer.window.add_plugin_dock_widget('napari', 'Features table widget')

if __name__ == '__main__':
    napari.run()
