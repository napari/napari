import napari
import skimage.data as data
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
    viewer = napari.Viewer(ndisplay=2)
    # viewer.axes.visible = True
    # viewer.window.qt_viewer.viewer.axes.visible = Tr
    image_data = data.astronaut()
    (ny, nx, _) = image_data.shape

    layer = viewer.add_image(image_data)


    # volume_node = viewer.window.qt_viewer.layer_to_visual[image_layer].node
    # points_layer = viewer.add_points(points_data, face_color='cornflowerblue')


    def update_x_range(event):
        x_range = event
        print('x: ', x_range)

        layer._set_bbox_lim(x_range, 1)


    def update_y_range(event):
        y_range = event
        print('y: ', y_range)

        layer._set_bbox_lim(y_range, 0)


    x_lim_slider = QHRangeSlider((0, nx - 1), data_range=(0, nx - 1))
    x_lim_slider.valuesChanged.connect(update_x_range)

    y_lim_slider = QHRangeSlider((0, ny - 1), data_range=(0, ny - 1))
    y_lim_slider.valuesChanged.connect(update_y_range)


    viewer.window.add_dock_widget(x_lim_slider, area='right')
    viewer.window.add_dock_widget(y_lim_slider, area='right')

