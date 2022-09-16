"""
bbox annotator
==============

.. tags:: gui
"""

from magicgui.widgets import ComboBox, Container
import napari
import numpy as np
import pandas as pd
from skimage import data


# set up the categorical annotation values and text display properties
box_annotations = ['person', 'sky', 'camera']
text_feature = 'box_label'
features = pd.DataFrame({
    text_feature: pd.Series([], dtype=pd.CategoricalDtype(box_annotations))
})
text_color = 'green'
text_size = 20


# create the GUI for selecting the values
def create_label_menu(shapes_layer, label_feature, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        a napari shapes layer
    label_feature : str
        the name of the shapes feature to use the displayed text
    labels : List[str]
        list of the possible text labels values.

    Returns
    -------
    label_widget : magicgui.widgets.Container
        the container widget with the label combobox
    """
    # Create the label selection menu
    label_menu = ComboBox(label='text label', choices=labels)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu():
        """This is a callback function that updates the label menu when
        the default features of the Shapes layer change
        """
        new_label = str(shapes_layer.feature_defaults[label_feature][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    shapes_layer.events.feature_defaults.connect(update_label_menu)

    def set_selected_features_to_default():
        """This is a callback that updates the feature values of the currently
        selected shapes. This is a side-effect of the deprecated current_properties
        setter, but does not occur when modifying feature_defaults."""
        indices = list(shapes_layer.selected_data)
        default_value = shapes_layer.feature_defaults[label_feature][0]
        shapes_layer.features[label_feature][indices] = default_value
        shapes_layer.events.features()

    shapes_layer.events.feature_defaults.connect(set_selected_features_to_default)
    shapes_layer.events.features.connect(shapes_layer.refresh_text)

    def label_changed(value: str):
        """This is a callback that update the default features on the Shapes layer
        when the label menu selection changes
        """
        shapes_layer.feature_defaults[label_feature] = value
        shapes_layer.events.feature_defaults()

    label_menu.changed.connect(label_changed)

    return label_widget


# create a stack with the camera image shifted in each slice
n_slices = 5
base_image = data.camera()
image = np.zeros((n_slices, base_image.shape[0], base_image.shape[1]), dtype=base_image.dtype)
for slice_idx in range(n_slices):
    shift = 1 + 10 * slice_idx
    image[slice_idx, ...] = np.pad(base_image, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]


# create a viewer with a fake t+2D image
viewer = napari.view_image(image)

# create an empty shapes layer initialized with
# text set to display the box label
text_kwargs = {
    'string': text_feature,
    'size': text_size,
    'color': text_color
}
shapes = viewer.add_shapes(
    face_color='black',
    features=features,
    text=text_kwargs,
    ndim=3
)

# create the label section gui
label_widget = create_label_menu(
    shapes_layer=shapes,
    label_feature=text_feature,
    labels=box_annotations
)
# add the label selection gui to the viewer as a dock widget
viewer.window.add_dock_widget(label_widget, area='right', name='label_widget')

# set the shapes layer mode to adding rectangles
shapes.mode = 'add_rectangle'

if __name__ == '__main__':
    napari.run()
