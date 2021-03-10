import os

os.environ.setdefault('NAPARI_OCTREE', '1')

import dask.array as da
import napari


ndim = 2
data = da.random.randint(0,255, (65536,) * ndim, chunks=(256,) * ndim, dtype='uint8')

viewer = napari.Viewer()
viewer.add_image(data, contrast_limits=[0, 255])
# To turn off grid lines
#viewer.layers[0].display.show_grid = False
viewer.camera.zoom = 0.75
napari.run()
