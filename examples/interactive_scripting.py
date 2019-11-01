import numpy as np
import napari
import time


with napari.gui_qt():
    # create the viewer with an image
    data = np.random.random((512, 512))
    viewer = napari.Viewer()
    layer = viewer.add_image(data)

    update_period = 0.1

    def layer_update(update_interval, kwargs):
        # number of times to update

        for k in range(kwargs['number_of_times']):  # Usage of keyword arguments
            time.sleep(update_interval[0])  # Usage of ordinary arguments

            dat = np.random.random((512, 512))
            layer.data = dat

            # check that data layer is properly assigned and not blocked?
            while layer.data.all() != dat.all():
                layer.data = dat

    update_thread = viewer.update(layer_update, update_period, number_of_times=100)
