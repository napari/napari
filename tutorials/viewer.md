# Napari Viewer Tutorial

Welcome to Napari Viewer Tutorial. In this tutorial we will be 
covering listed topics about the viewer:

- Starting the viewer
- Changing between themes of viewer
- Reordering layers on the viewer
- Custom keybinding 

### Starting the viewer

First start with importing `napari` and 
`skimage.data` for sample data.

```python
import napari
from skimage import data
```

Continue with opening `napari.qui_qt` and open the viewer with the sample data. 
Here `data.camaera()` returns type of **numpy.ndarray**. So basically you can 
read any image into such array type and open with napari viewer easily. By 
now napari does not provide any I/O help and expecting numpy array to work on.

```python
with napari.qui_qt():
    viewer = napari.view(data.camera())
```

### Changing the theme of the viewer

Currently, napari comes with two different themes and `dark` is the default. In 
case you want to change this, just update `theme` property of the viewer. 
Likewise you can set it back to `dark` theme.

```python
viewer.theme = 'light'
```

### Reordering layers on the viewer

Napari supports having multiple layers within a single viewer. One can superimpose
layers and can manipulate the ordering of the layers.

```python
with napari.qui_qt():
    viewer = napari.view(photographer=data.camera(),
                         coins=data.coins(),
                         moon=data.moon())
                         
    # swap layer order
    viewer.layers['photographer', 'moon'] = viewer.layers['moon', 'photographer']
```

### Custom keybinding

To be filled
