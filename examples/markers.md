```python
%gui qt5
```

```python
import napari_gui as gui
import numpy as np
```

```python
# Create a viewer with a random 5D image (e.g., [x, y, z, c, t])
rand_im = gui.imshow(np.random.rand(500, 500, 5, 3, 2))
```

```python
# ? gives usage instructions for add_markers
rand_im.add_markers?
```

```python
# Create a list of marker coords
marker_list = np.array([[50, 30, 0, 0, 0],
                        [20, 20, 0, 0, 0],
                        [40, 90, 0, 0, 0],
                        [50, 40, 1, 1, 1],
                        [70, 30, 4, 2, 1]])

# Create the layer with markers
rand_im.add_markers(marker_list)
```

```python
# We can also change the marker styling by updating the layer properties
print(rand_im.layers)

rand_im.layers[1].size = np.array([20, 20, 20, 5, 5])
rand_im.layers[1].symbol= '+'
```

```python
# Sizes can also be set with lists
rand_im.layers[1].size = [10, 10, 10, 5, 5]
```

```python
# We can update the marker coordinates via the data property
marker_list2 = np.array([[400, 30, 0, 0, 0],
                        [75, 200, 0, 0, 0],
                        [33, 90, 0, 0, 0],
                        [99, 40, 1, 1, 1],
                        [50, 30, 4, 2, 1]])

rand_im.layers[1].data = marker_list2
```

```python
# Similarly, we can update the marker coordinates via the coords property
marker_list3 = np.array([[50, 30, 0, 0, 0],
                        [10, 200, 0, 0, 0],
                        [33, 90, 0, 0, 0],
                        [99, 40, 1, 1, 1],
                        [50, 30, 4, 2, 1]])

rand_im.layers[1].coords = marker_list3
```

```python

```
