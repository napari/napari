from napari.settings import get_settings
import time
import napari
from napari._qt.qthreading import thread_worker
from skimage import data
from skimage.morphology import ball, octahedron
import matplotlib.pyplot as plt


def make_screenshot(viewer):
    img = viewer.screenshot(canvas_only=True, flash=False)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


#get_settings().application.window_size = (600, 600) # I will delete this line before merge
get_settings().application.window_position = (900, 300) # hopefully better option to do this
viewer = napari.Viewer()
viewer.window.resize(600, 600) 
#viewer.window.position(600, 600) # is it possible that this will replace get_settings().application.window_position = (900, 300)?

viewer.window.qt_viewer.dockLayerControls.toggleViewAction().trigger()
# in napari 0.4.13 it will become private:
# viewer.window._qt_viewer.dockLayerControls.toggleViewAction().trigger()
# but maybe a better public method will come for this

viewer.theme = "light"
viewer.dims.ndisplay = 3
viewer.axes.visible = True
viewer.axes.colored = False
viewer.axes.labels = False
viewer.text_overlay.visible = True
viewer.text_overlay.text = "Hello World!"

myblob = data.binary_blobs(
    length=200, volume_fraction=0.1, blob_size_fraction=.3,  n_dim=3, seed=42)
myoctahedron = octahedron(100)
myball = ball(100)

files = {
    "blob": myblob,
    "ball": myball,
    "octahedron": myoctahedron,
}

viewer.add_labels(myball, name='result') # better way to init empty labels?
viewer.camera.angles = (19, -33, -121)
# viewer.camera.zoom = 1.7 why not here?


@thread_worker
def loop_run():
    for key in files:
        time.sleep(0.5)
        image = files[key]
        yield (image, key)


def update_layer(image_text_tuple):
    image, text = image_text_tuple
    viewer.layers['result'].data = image
    viewer.text_overlay.text = text
    viewer.camera.zoom = 1.7

    make_screenshot(viewer)


worker = loop_run()
worker.yielded.connect(update_layer)
worker.start()