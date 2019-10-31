import numpy as np
import napari
import time


with napari.gui_qt():
    # create the viewer with an image
    data = np.random.random((512, 512))
    viewer = napari.Viewer()
    layer = viewer.add_image(data)

    def layer_update():
        # number of times to update
        number_of_times = 100

        # time between image assignment
        update_interval = 0.1

        for k in range(number_of_times):

            time.sleep(update_interval)

            dat = np.random.random((512, 512))
            layer.data = dat

            # check that data layer is properly assigned and not blocked?
            while layer.data.all() != dat.all():
                layer.data = dat

    update_thread = viewer.update_viewer(layer_update)
