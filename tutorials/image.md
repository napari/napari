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
