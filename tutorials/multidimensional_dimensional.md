**Under construction**

# Napari nD Tutorial

Welcome to Napari nD Tutorial. In this tutorial we will be
covering listed topics about the viewer:

- Ways to start viewer with nD data
- Slicing
- Min-Max filtering over dimensions
...

### Ways to start viewer with nD data

It has always been a great challenge to visualize and browse high dimensional
data. Napari handles nD data with slicing mainly. As we discussed in layers
tutorial, one can read any nD data into **numpy.ndarray** type and pass it to
`napari.view()` method to start napari with added nD layer. By default, napari
will apply slicing to higher dimensional data.

```python
import numpy as np
from skimage import data
import napari

with napari.gui_qt():
    # create fake 3d data
    blobs = np.stack([data.binary_blobs(length=128, blob_size_fraction=0.05,
                                        n_dim=3, volume_fraction=f)
                     for f in np.linspace(0.05, 0.5, 10)], axis=-1)
    # add data to the viewer
    viewer = napari.view(blobs.astype(float))
```

### Slicing

...

### Min-Max filtering over dimensions

...
