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

```{tags} gui
```

# Creating reproducible screenshots with a viewer loop

This example captures images in three dimensions for multiple samples.  
This can be e.g. useful when one has dozens of ct scans and wants to visualize them for a quick overview with napari but does not want to load them one by one.  
Reproducibility is achieved by defining exact frame width and frame height.  

+++

The first cell takes care of the imports and data initializing, in this case a blob, a ball and an octahedron.  

```{code-cell} ipython3
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
    
myblob = data.binary_blobs(
    length=200, volume_fraction=0.1, blob_size_fraction=0.3, n_dim=3, seed=42
)
myoctahedron = octahedron(100)
myball = ball(100)

# store the variables in a dict with the image name as key.
data_dict = {
    "blob": myblob,
    "ball": myball,
    "octahedron": myoctahedron,
}
```

Now, the napari viewer settings can be adjusted programmatically, such as 3D rendering methods, axes visible, color maps, zoom level, and camera orientation.    
Every plot will have these exact settings, while only one napari viewer instance is needed.  
After setting these parameters, one should not make changes with the mouse in the napari viewer anymore, as this would rule out the reproducibility.

```{code-cell} ipython3
viewer = napari.Viewer()
viewer.window.resize(900, 600)

viewer.theme = "light"
viewer.dims.ndisplay = 3
viewer.axes.visible = True
viewer.axes.colored = False
viewer.axes.labels = False
viewer.text_overlay.visible = True
viewer.text_overlay.text = "Hello World!"

# Not yet implemented, but can be added as soon as this feature exisits (syntax might change): 
# viewer.controls.visible = False

viewer.add_labels(myball, name="result" , opacity=1)
viewer.camera.angles = (19, -33, -121)
viewer.camera.zoom = 1.3
```

Next, the loop run is defined.  
The `loop_run` function reads new `image_data` and the corresponding `image_name` and yields them to napari.     
The `update_layer` function gives instructions how to process the yielded data in the napari viewer.

```{code-cell} ipython3
@thread_worker
def loop_run():
    for image_name in data_dict: 
        time.sleep(0.5)
        image_data = data_dict[image_name]
        yield (image_data, image_name)

def update_layer(image_text_tuple):
    image, text = image_text_tuple
    viewer.layers["result"].data = image
    viewer.text_overlay.text = text
    make_screenshot(viewer)
```

And finally, the loop is executed:

```{code-cell} ipython3
worker = loop_run()
worker.yielded.connect(update_layer)
worker.start()
```

```{code-cell} ipython3

```
