# Napari Layers Tutorial

Welcome to Napari Layers Tutorial. In this tutorial we will be 
covering listed types of Layers:

- Image Layer
- Label Layer
- Marker Layer
- Vector Layer
- Shape Layer
- Pyramid Layer

### Image Layer

Image layer is the most commonly used type of layer. By now, napari 
does not provide any I/O utilities for any specific image format. Hence, 
it is users responsibility to read data in a **numpy.ndarray** and pass 
to napari. After reading such array, just passing it to `napari.view()`
method will be enough to start napari. In the example below, `data.camera()`
returns such numpy array and starts napari with that image layer.

```python
import napari
from skimage import data

with napari.qui_qt():
    viewer = napari.view(data.camera())
```


### Label Layer

```python
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
import napari


with napari.gui_qt():
    image = data.coins()[50:-50, 50:-50]

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    # initialise viewer with coins image
    viewer = napari.view(coins=image, multichannel=False)
    viewer.layers[0].colormap = 'gray'

    # add the labels
    label_layer = viewer.add_labels(label_image, name='segmentation')
```

### Marker Layer

```python
import numpy as np
from skimage import data
import napari

with napari.gui_qt():
    # set up viewer
    viewer = napari.Viewer()
    viewer.add_image(data.camera()))
    
    # create three xy coordinates
    points = np.array([[100, 100], [200, 200], [333, 111]])
    
    # specify three sizes
    size = np.array([10, 30, 20])
    
    # add them to the viewer
    points = viewer.add_points(points, size=size)
```


### Vector Layer

...

### Shape Layer

...

### Pyramid Layer

...
