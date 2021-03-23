import napari
import skimage.data as data
from magicgui.widgets import Container
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
    # viewer.axes.visible = True
    # viewer.window.qt_viewer.viewer.axes.visible = True
    data = np.random.random((100, 200, 300))
    layer = viewer.add_image(data)

    def update_x_range(event):
        x_range = event
        print(x_range)

        layer._set_xlim(np.asarray(x_range))
    
    def update_y_range(event):
        y_range = event
        print(y_range)

        layer._set_ylim(np.asarray(y_range))

    def update_z_range(event):
        z_range = event
        print(z_range)

        layer._set_zlim(np.asarray(z_range))

    x_lim_slider = QHRangeSlider((0, 299), data_range=(0, 299))
    x_lim_slider.valuesChanged.connect(update_x_range)

    y_lim_slider = QHRangeSlider((0, 199), data_range=(0, 199))
    y_lim_slider.valuesChanged.connect(update_y_range)

    z_lim_slider = QHRangeSlider((0, 99), data_range=(0, 99))
    z_lim_slider.valuesChanged.connect(update_z_range)

    viewer.window.add_dock_widget(x_lim_slider, area='right')
    viewer.window.add_dock_widget(y_lim_slider, area='right')
    viewer.window.add_dock_widget(z_lim_slider, area='right')
