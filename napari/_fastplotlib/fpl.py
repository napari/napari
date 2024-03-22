import imageio.v3 as iio

import napari
from napari._fastplotlib import FastplotlibCanvas, FastplotlibImageLayer
from napari.layers import Image

# create napari viewer
viewer = napari.Viewer()

im = iio.imread("imageio:camera.png")

# create napari image
napari_image = Image(data=im, name="iio camera", colormap="plasma")

# create corresponding fpl image layer
fpl_image = FastplotlibImageLayer(napari_layer=napari_image)

# create fpl widget
fpl_canvas = FastplotlibCanvas()

# add widget as dock widget to napari viewer
viewer.window.add_dock_widget(fpl_canvas)

# add image graphic to plot
fpl_canvas._plot.add_graphic(fpl_image.image_graphic)

# add napari image to viewer
viewer.add_layer(napari_image)

# flip camera
fpl_canvas._plot.camera.world.scale_y *= -1

if __name__ == "__main__":
    napari.run()
