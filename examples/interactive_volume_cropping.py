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

# add the volume
with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.window.qt_viewer.viewer.axes.visible = True
    data = np.random.random((100, 200, 300))
    layer = viewer.add_image(data)

    # def update_x_range(event):
    #     x_range = event
    #     print(x_range)

    #     volume_node = viewer.window.qt_viewer.layer_to_visual[layer].node
    #     volume_node.bounding_box_zlim = x_range

    # slider_widget = QHRangeSlider((0, 128), data_range=(0, 128))
    # slider_widget.valuesChanged.connect(update_x_range)


    # viewer.window.add_dock_widget(slider_widget)
