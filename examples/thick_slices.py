import numpy as np
import napari
from magicgui import magicgui


points = np.random.rand(100, 3) * 100

viewer = napari.Viewer()
layer = viewer.add_points(points)

@magicgui(
    auto_call=True,
    thickness=dict(widget_type='Slider', min=1, max=100)
)
def change_thickness(viewer: napari.Viewer, thickness: float):
    viewer.dims.thickness_slices = thickness, 1, 1


viewer.window.add_dock_widget(change_thickness)
napari.run()
