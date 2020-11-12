# demonstrate how to find the vispy 3D Texture for a layer
# and how to update part of it

import numpy as np
import napari
from napari.qt.threading import thread_worker
import time
import tifffile


# vol = np.random.randint(0, 255, size=(nz, ny, nx), dtype=np.uint8)
vol = tifffile.imread("ovarioles_small.tif")
ones = np.ones_like(vol)
zeros = np.zeros_like(vol)
inverted = np.clip(vol + 80, 0, 255)
nz, ny, nx = vol.shape


@thread_worker
def pulse():
    count = -2
    while True:
        time.sleep(0.1)
        count += 2
        yield count % nz


with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3, title="livevolume")
    viewer.add_image(zeros, name="Live Acquisition")

    vispy_layer = viewer.window.qt_viewer.layer_to_visual[viewer.layers[0]]
    volume = vispy_layer._volume_node
    texture = volume._tex
    print(type(texture))
    # texture.set_data(ones[:3,...], offset=(3,0,0))

    def update_slice(zslice):
        texture.set_data(
            inverted[zslice : zslice + 2, ...], offset=(zslice, 0, 0)
        )
        print(zslice)
        if zslice > 1:
            texture.set_data(
                vol[zslice - 2 : zslice, ...], offset=(zslice - 2, 0, 0)
            )
        else:
            texture.set_data(vol[nz - 2 : nz, ...], offset=(nz - 2, 0, 0))
        volume.update()

    worker = pulse()
    worker.yielded.connect(update_slice)
    worker.start()
