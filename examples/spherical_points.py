import napari
import numpy as np


np.random.seed()

pts = np.random.rand(100, 3) * 100
colors = np.random.rand(100, 3)
sizes = np.random.rand(100) * 20 + 10

viewer = napari.Viewer(ndisplay=3)
pts_layer = viewer.add_points(
    pts,
    face_color=colors,
    size=sizes,
    shading='spherical',
    edge_width=0,
)

# antialias is currently a bit broken, which is especially bad in 3D
# we can use a private attribute for now (beware, this is not public API!)
pts_layer._antialias = 0

viewer.reset_view()

napari.run()
