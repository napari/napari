"""
Display a 3D volume and the fps label
"""
import numpy as np
import napari


def update_fps(fps):
    """Update fps"""
    viewer.label.text = "%1.1f FPS" % fps

viewer = napari.Viewer()
viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
viewer.label.visible = True
viewer.window.qt_viewer.canvas.measure_fps(callback=update_fps)

napari.run()