### Vector Layer

Vector layer enables one to draw arbitrary vectors on an image layer.
With help of vector layer napari able to provide vectoral marking. A 
brief example given below:

```python
import napari
from skimage import data
import numpy as np


with napari.gui_qt():
    # create the viewer and window
    viewer = napari.Viewer()

    layer = viewer.add_image(data.camera(), name='photographer')
    layer.colormap = 'gray'

    # sample vector coord-like data
    n = 1000
    pos = np.zeros((n, 2, 2), dtype=np.float32)
    phi_space = np.linspace(0, 4 * np.pi, n)
    radius_space = np.linspace(0, 100, n)

    # assign x-y position
    pos[:, 0, 0] = radius_space * np.cos(phi_space) + 350
    pos[:, 0, 1] = radius_space * np.sin(phi_space) + 256

    # assign x-y projection
    pos[:, 1, 0] = 2 * radius_space * np.cos(phi_space)
    pos[:, 1, 1] = 2 * radius_space * np.sin(phi_space)

    # add the vectors
    layer = viewer.add_vectors(pos, width=0.4)
```
