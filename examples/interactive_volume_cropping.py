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
    
    nx, ny, nz = 100, 200, 300
    image_data = np.random.random((nz, ny, nx))
    points_data = np.array([[0, 0, 0],
                            [0, 0, nx],
                            [0, ny, 0],
                            [0, ny, nx],
                            [nz, 0, 0],
                            [nz, 0, nx],
                            [nz, ny, 0],
                            [nz, ny, nx]
                        ])
    layer = viewer.add_image(image_data)
    # volume_node = viewer.window.qt_viewer.layer_to_visual[image_layer].node
    points_layer = viewer.add_points(points_data, face_color='cornflowerblue')

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

    x_lim_slider = QHRangeSlider((0, nx - 1), data_range=(0, nx - 1))
    x_lim_slider.valuesChanged.connect(update_x_range)

    y_lim_slider = QHRangeSlider((0, ny - 1), data_range=(0, ny - 1))
    y_lim_slider.valuesChanged.connect(update_y_range)

    z_lim_slider = QHRangeSlider((0, nz - 1), data_range=(0, nz - 1))
    z_lim_slider.valuesChanged.connect(update_z_range)

    viewer.window.add_dock_widget(x_lim_slider, area='right')
    viewer.window.add_dock_widget(y_lim_slider, area='right')
    viewer.window.add_dock_widget(z_lim_slider, area='right')

