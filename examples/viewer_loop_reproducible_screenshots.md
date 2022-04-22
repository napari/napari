---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import napari
napari.__version__
```

```{code-cell} ipython3
#header: imports and define viewer layout
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


get_settings().application.window_position = (600, 600)
viewer = napari.Viewer()
viewer.window.resize(900, 600)

viewer.theme = "light"
viewer.dims.ndisplay = 3
viewer.axes.visible = True
viewer.axes.colored = False
viewer.axes.labels = False
viewer.text_overlay.visible = True
viewer.text_overlay.text = "Hello World!"
```

```{code-cell} ipython3
# define data and update loop 
myblob = data.binary_blobs(
    length=200, volume_fraction=0.1, blob_size_fraction=0.3, n_dim=3, seed=42
)
myoctahedron = octahedron(100)
myball = ball(100)

files = {
    "blob": myblob,
    "ball": myball,
    "octahedron": myoctahedron,
}

viewer.add_labels(myball, name="result")
viewer.camera.angles = (19, -33, -121)
viewer.camera.zoom = 1.3


@thread_worker
def loop_run():
    for key in files:
        time.sleep(0.5)
        image = files[key]
        yield (image, key)


def update_layer(image_text_tuple):
    image, text = image_text_tuple
    viewer.layers["result"].data = image
    viewer.layers["result"].opacity = 1
    viewer.text_overlay.text = text
    make_screenshot(viewer)
```

```{code-cell} ipython3
# start loop
worker = loop_run()
worker.yielded.connect(update_layer)
worker.start()
```

```{code-cell} ipython3

```
