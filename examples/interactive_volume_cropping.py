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
    
    nx, ny, nz, nt = 28, 56, 128, 10
    image_data = np.random.random((nt, nz, ny, nx))
    # points_data = np.array([[0, 0, 0],
    #                         [0, 0, nx],
    #                         [0, ny, 0],
    #                         [0, ny, nx],
    #                         [nz, 0, 0],
    #                         [nz, 0, nx],
    #                         [nz, ny, 0],
    #                         [nz, ny, nx]
    #                     ])
    layer = viewer.add_image(image_data)
    # volume_node = viewer.window.qt_viewer.layer_to_visual[image_layer].node
    # points_layer = viewer.add_points(points_data, face_color='cornflowerblue')

    def update_t_range(event):
        t_range = event
        print('t: ', t_range)

        layer._set_bbox_lim(t_range, 0)

    def update_x_range(event):
        x_range = event
        print('x: ', x_range)

        layer._set_bbox_lim(x_range, 3)
    
    def update_y_range(event):
        y_range = event
        print('y: ', y_range)

        layer._set_bbox_lim(y_range, 2)

    def update_z_range(event):
        z_range = event
        print('z: ', z_range)

        layer._set_bbox_lim(z_range, 1)

    t_lim_slider = QHRangeSlider((0, nt - 1), data_range=(0, nt - 1))
    t_lim_slider.valuesChanged.connect(update_t_range)

    x_lim_slider = QHRangeSlider((0, nx - 1), data_range=(0, nx - 1))
    x_lim_slider.valuesChanged.connect(update_x_range)

    y_lim_slider = QHRangeSlider((0, ny - 1), data_range=(0, ny - 1))
    y_lim_slider.valuesChanged.connect(update_y_range)

    z_lim_slider = QHRangeSlider((0, nz - 1), data_range=(0, nz - 1))
    z_lim_slider.valuesChanged.connect(update_z_range)

    viewer.window.add_dock_widget(t_lim_slider, area='right')
    viewer.window.add_dock_widget(x_lim_slider, area='right')
    viewer.window.add_dock_widget(y_lim_slider, area='right')
    viewer.window.add_dock_widget(z_lim_slider, area='right')

