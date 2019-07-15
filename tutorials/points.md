**Under construction**

### Point Layer

Point layer is very useful to mark certain points on a layer.
A brief usage is shown in below:

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

![image](../resources/screenshot-add-points.png)

One can also get coordinates of points with `points.coords`.
