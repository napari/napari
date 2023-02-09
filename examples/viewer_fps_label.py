"""
Viewer FPS label
================

Display a 3D volume and the fps label.

.. tags:: experimental
"""
import numpy as np

import napari


def update_fps(fps):
    """Update fps."""
    viewer.text_overlay.text = f"{fps:1.1f} FPS"


viewer = napari.Viewer()
viewer.add_image(np.random.random((5, 5, 5)), colormap='red', opacity=0.8)
viewer.text_overlay.visible = True
viewer.window.qt_viewer.canvas.measure_fps(callback=update_fps)

if __name__ == '__main__':
    napari.run()
