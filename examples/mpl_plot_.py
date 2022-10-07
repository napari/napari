"""
Matplotlib plot
===============

.. tags:: gui
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas

import napari

# create image
x = np.linspace(0, 5, 256)
y = np.linspace(0, 5, 256)[:, np.newaxis]
img = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

# add it to the viewer
viewer = napari.view_image(img, colormap='viridis')
layer = viewer.layers[-1]

# create mpl figure with subplots
mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
(line,) = ax.plot(layer.data[123])  # linescan through the middle of the image

# add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))


# connect a callback that updates the line plot when
# the user clicks on the image
@layer.mouse_drag_callbacks.append
def profile_lines_drag(layer, event):
    try:
        line.set_ydata(layer.data[int(event.position[0])])
        line.figure.canvas.draw()
    except IndexError:
        pass


if __name__ == '__main__':
    napari.run()
