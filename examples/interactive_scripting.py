import numpy as np
import napari
from napari._qt.threading import worker_factory
import time


with napari.gui_qt():
    # create the viewer with an image
    data = np.random.random((512, 512))
    viewer = napari.Viewer()
    layer = viewer.add_image(data)

    def layer_update(*, update_period, num_updates):
        yield
        # number of times to update
        for k in range(num_updates):
            time.sleep(update_period)

            dat = np.random.random((512, 512))
            layer.data = dat

            # check that data layer is properly assigned and not blocked?
            while layer.data.all() != dat.all():
                layer.data = dat
            yield

    worker_factory(layer_update, update_period=0.05, num_updates=100)
