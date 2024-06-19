"""
Viewer FPS label
================

Display a 3D volume and the fps label.
"""
import logging

import numpy as np

import napari

logging.basicConfig(level=logging.INFO)


viewer = napari.Viewer(ndisplay=3)
image_layer = viewer.add_image(
    np.random.random((700, 700, 500)),
    colormap='magenta',
    opacity=0.8,
    blending='additive',
)
image_layer2 = viewer.add_image(
    np.random.random((700, 700, 500)),
    translate=(700, 0, 0),
    colormap='green',
    opacity=0.8,
    blending='additive',
)

viewer.text_overlay.visible = True

visual = viewer.window._qt_window._qt_viewer.layer_to_visual[image_layer]
node = visual._layer_node.get_node(3)
print(node.relative_step_size)


def on_draw(event=None):
    """Display the current frame rate and step size as a text overlay"""
    #print("on draw called")
    fps_monitor = viewer.window._qt_window._qt_viewer._fps_monitor
    fps = viewer.window._qt_window._qt_viewer.canvas._scene_canvas.fps
    fps_valid = fps_monitor.valid

    viewer.text_overlay.text = f"{fps:1.1f} FPS, valid: {fps_valid}, step: {node.relative_step_size}"


viewer.window.qt_viewer.canvas._scene_canvas.events.draw.connect(on_draw)


if __name__ == '__main__':
    napari.run()
