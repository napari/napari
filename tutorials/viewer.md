# napari viewer tutorial

Welcome to the tutorial on the **napari** viewer. This tutorial assumes you have already installed **napari** and know how to launch the viewer. For help with installation see our [installation](installation.md) tutorial. For help getting started with the viewer see [getting started](getting_started.md) tutorial. This tutorial will teach you about the **napari** viewer, including how to use it on your screen and how the data within it is organised. At the end of the tutorial you should understand the both the layout of the viewer on the screen and the data inside of it.


## launching the viewer
As discussed in [getting started](getting_started.md) tutorial the napari viewer can be launched from the command-line, a python script, or a jupyter notebook / iPython console. All three methods launch the same viewer and anything related to the interacting with the viewer on the screen applies equally to all of them. We will use the iPython console in these examples as it gives us the most control when interacting with the viewer, but the same syntax can be used in python scripts.

Let's get stated by launching a viewer with a simple 2D image.

The fasted way to get the viewer open and throw an image up on the screen is using the `napari.view` method:

```python
%gui qt5
from skimage import data
import napari

viewer = napari.view(data.astronaut())
```
Calling `napari.view` will return a `Viewer` object that is the main object inside **napari**. All the data you add to **napari** will be stored inside the `Viewer` object.

You can also create an empty `Viewer` directly and then start adding images to it. This approach can be more flexible for complex work flows, and will allow you to add other types of data like `points` and `shapes`.

```python
%gui qt5
from skimage import data
import napari

viewer = napari.Viewer()
viewer.add_image(data.astronaut())
```

After running either of those two commands you should now be able to see the photograph of the astronaut in the a **napari** viewer as shown below

![image](resources/viewer_astronaut.png)

Both the `view` and the `add_image` methods accept any numpy-array like object as an input, including n-dimensional arrays. For more information on adding images to the viewer see the [image layer](image.md) tutorial. Now we will continue exploring the rest of the viewer.

**Under construction**


## layout of the viewer

The viewer is organized into ....


## reordering layers on the viewer

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

## custom keybinding

To be filled in


## changing the theme of the viewer

Currently, napari comes with two different themes and `dark` is the default. In
case you want to change this, just update `theme` property of the viewer.
Likewise you can set it back to `dark` theme.

```python
viewer.theme = 'light'
```

## next steps

To learn more about the different layers that **napari** supports checkout some more of our tutorials listed below. The [image layer](image.md) tutorial is a great one to try next as viewing images is a fundamental part of what **napari** is about.

## all tutorials

- [installing napari](installation.md)
- [getting started tutorial](getting_started.md)
- [napari viewer tutorial](viewer.md)
- [multidimensional tutorial](multidimensional_dimensional.md)
- [image layer tutorial](image.md)
- [labels layer tutorial](labels.md)
- [points layer tutorial](points.md)
- [shapes layer tutorial](shapes.md)
- [pyramid layer tutorial](pyramid.md)
- [vectors layer tutorial](vectors.md)
