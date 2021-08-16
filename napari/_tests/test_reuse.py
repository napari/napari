import numpy as np

import napari


def test_closed_viewer_ok():
    v = napari.Viewer()
    v.close()
    v.add_points()
    v.add_image(np.random.rand(4, 4))
    v.show()
