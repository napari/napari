# Setup

```python
%gui qt5
```

```python
import napari_gui as gui
import numpy as np
```

## Display a single image

```python
viewer = gui.imshow(np.random.rand(500, 500))
```

### Add a new layer

```python
viewer.imshow(np.random.rand(500, 500, 3))
```

### Add a third layer

```python
viewer.add_image(np.random.rand(500, 500, 4), meta=dict(itype='rgb'))
```

### Add a marker layer

```python
marker_list = np.array([[100, 100], [200, 200], [333, 111]])
viewer.add_markers(marker_list, size=20)
```

### Reorder layers

```python
viewer.layers.swap(3,1)
```

```python
viewer.layers.reorder([2,1,3,0])
```

### Remove first layer

```python
del viewer.layers[0]
```

## Add a multidimensional layer

```python
h = 32
w = 32
d = 64
b = 64
C, Z, Y, X = np.ogrid[-2.5:2.5:h * 1j, -2.5:2.5:w * 1j, -2.5:2.5:d * 1j, -2.5:2.5:b * 1j]
image = np.empty((h, w, d, b), dtype=np.float32)
image[:] = np.exp(- X ** 2 - Y ** 2 - Z ** 2 - C ** 2)

layer = viewer.imshow(image)
```

```python
layer.scale = 500 / 32, 500 / 32
```

```python
layer.cmap = 'blues'
```

```python
layer.interpolation = 'spline36'
```
