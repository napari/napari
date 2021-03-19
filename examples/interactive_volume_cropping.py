import napari
import skimage.data as data
from magicgui.widgets import Container, FloatSlider
from napari._qt.widgets.qt_range_slider import QHRangeSlider
import numpy as np

blobs = np.asarray(
    [
        data.binary_blobs(length=128, volume_fraction=0.1, n_dim=3).astype(
            float
        )
    ]
)

points = np.array([[0,0,0],
                  [0,0,1],
                   [0,1,0],
                   [1,0,0],
                   [0,1,1],
                   [1,0,1],
                   [1,1,0],
                   [1,1,1]]) * 128
viewer = napari.Viewer(ndisplay=3)


# add the volume
with napari.gui_qt():
    layer = viewer.add_image(blobs)
    viewer.add_points(points, face_color='cornflowerblue')

    def update_x_range(event):
        x_range = event
        print(x_range)

        volume_node = viewer.window.qt_viewer.layer_to_visual[layer].node
        volume_node.xlims = x_range

    slider_widget = QHRangeSlider((0, 128), data_range=(0, 128))
    slider_widget.valuesChanged.connect(update_x_range)


    viewer.window.add_dock_widget(slider_widget)
